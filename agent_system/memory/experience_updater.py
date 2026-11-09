"""
LLM-based experience updater for ALFWorld experience banks.

The updater consumes contrasted rollout snippets and emits structured
state-aware experiences instead of generic experiences.
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI


class ExperienceUpdater:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_new_experiences_per_update: int = 3,
        max_completion_tokens: int = 2048,
        token_limit_param: str = "max_tokens",
    ):
        api_key = api_key or os.environ.get("EXPERIENCE_UPDATE_API_KEY")
        base_url = base_url or os.environ.get("EXPERIENCE_UPDATE_BASE_URL")
        model = model or os.environ.get("EXPERIENCE_UPDATE_MODEL")

        if not api_key or not base_url or not model:
            raise EnvironmentError(
                "ExperienceUpdater requires api_key, base_url, and model. Provide them "
                "via constructor args or EXPERIENCE_UPDATE_API_KEY, "
                "EXPERIENCE_UPDATE_BASE_URL, and EXPERIENCE_UPDATE_MODEL environment variables."
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self._audit_request_count = 0
        if os.environ.get("T063_REFLECTOR_AUDIT") == "1":
            self.client._client.event_hooks.setdefault("request", []).append(self._audit_request)
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.max_new_experiences_per_update = max_new_experiences_per_update
        self.token_limit_param = token_limit_param
        self.update_history = []

    def _audit_request(self, request):
        self._audit_request_count += 1

    def analyze_failures(
        self,
        failed_trajectories: List[Dict],
        current_experiences: Dict,
    ) -> List[Dict]:
        if not failed_trajectories:
            return []

        next_exp_idx = self._next_exp_index(current_experiences)
        domain = self._prompt_domain(current_experiences)
        if domain == "search":
            prompt = self._build_search_analysis_prompt(failed_trajectories, current_experiences, next_exp_idx)
            if any(traj.get("fork_id") for traj in failed_trajectories):
                old_description = "Each example below is a contrast on the SAME question and the SAME turn: the rollout that received the listed experiences failed, while the rollout without any experience succeeded. So the injected experience either misled the agent or failed to prevent a mistake."
                new_description = "Each example is a fork at one identical search state. LEFT omits only this turn's new experiences; RIGHT adds them. Both sides may retain earlier task experience. The actions shown are this turn's real actions; scores are terminal subtree success fractions. Delta=RIGHT-LEFT can show helped, harmed, or neutral evidence. Distill a reusable strategy from either sign without treating a noisy fork as certain causality."
                prompt = prompt.replace(old_description, new_description)
                prompt = prompt.replace("Generate 1-", "Generate 0-")
                prompt += "\nFor each object include experience_type: task or step. task is a reusable multi-turn solving strategy, usually from a root fork. step is an action or warning for a concrete state, usually from an internal fork. You may return an empty array and need not produce both types.\n"
                prompt = prompt.replace("Use the successful no-experience rollout to see what worked, and the injected experience to see what went wrong.", "Compare the two actions and subtree scores to identify a useful strategy or a harmful application of experience.")
        else:
            prompt = self._build_analysis_prompt(failed_trajectories, current_experiences, next_exp_idx)
        self.last_prompt_domain = domain

        started = time.perf_counter()
        request_count_before = self._audit_request_count
        parsed_count = 0
        self.last_update_succeeded = False
        request_succeeded = False
        try:
            request_kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                self.token_limit_param: self.max_completion_tokens,
            }
            response = self.client.chat.completions.create(**request_kwargs)
            request_succeeded = True
            raw_experiences = self._parse_experiences_response(response.choices[0].message.content)
            parsed_count = len(raw_experiences)
            self.last_update_succeeded = self.last_parse_succeeded
            if domain == "search":
                raw_experiences = self._postprocess_search_experiences(raw_experiences, failed_trajectories)
            reassigned = self._reassign_exp_ids(raw_experiences, next_exp_idx)
            self.update_history.append(
                {
                    "num_failures_analyzed": len(failed_trajectories),
                    "num_experiences_generated": len(reassigned),
                    "experience_ids": [s.get("experience_id") for s in reassigned],
                }
            )
            return reassigned[: self.max_new_experiences_per_update]
        except Exception as e:
            safe_error = str(e).replace(self.client.api_key, "[REDACTED]")
            print(f"[ExperienceUpdater] Error calling {self.model}: {safe_error}")
            return []
        finally:
            if os.environ.get("T063_REFLECTOR_AUDIT") == "1":
                attempts = self._audit_request_count - request_count_before
                print("T063_REFLECTOR_RESULT " + json.dumps({
                    "model": self.model,
                    "request_succeeded": request_succeeded,
                    "parsed_experiences": parsed_count,
                    "http_attempts": attempts,
                    "retry_count": max(0, attempts - 1),
                    "seconds": time.perf_counter() - started,
                }), flush=True)

    # ------------------------------------------------------------------ #
    # Prompt construction                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _iter_existing_experiences(current_experiences: Dict):
        """Yield experiences from legacy and V3 banks without double-counting them.

        The initial state-aware bank stored items under ``experiences``.  V3
        stores them in ``task_experiences`` and ``step_experiences`` instead, with a
        legacy ``experience_id`` retained on step experiences.  Experience evolution has
        to understand both forms because an updated V3 bank is passed straight
        back to this updater on the next training iteration.
        """
        if not isinstance(current_experiences, dict):
            return

        seen = set()
        for pool_name in ("experiences", "task_experiences", "step_experiences"):
            pool = current_experiences.get(pool_name, []) or []
            if not isinstance(pool, list):
                continue
            for experience in pool:
                if not isinstance(experience, dict):
                    continue
                # A migration/adapter may expose the same object in two pools.
                # Deduplicate by its stable identifier and fall back to object
                # identity for identifier-free entries.
                identifier = str(experience.get("experience_id") or "")
                key = (identifier, pool_name) if not identifier else (identifier,)
                if key in seen:
                    continue
                seen.add(key)
                yield pool_name, experience

    @classmethod
    def _existing_experience_ids(cls, current_experiences: Dict) -> set[str]:
        """Return all experience IDs present in any supported bank schema."""
        ids = set()
        for _, experience in cls._iter_existing_experiences(current_experiences):
            for field in ("experience_id",):
                value = experience.get(field)
                if value is not None:
                    value = str(value)
                    if value:
                        ids.add(value)
        return ids

    def _next_exp_index(self, current_experiences: Dict) -> int:
        max_idx = 0
        pattern = re.compile(r"^exp_(\d+)$")
        for experience_id in self._existing_experience_ids(current_experiences):
            m = pattern.match(experience_id)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
        return max_idx + 1

    def _reassign_exp_ids(self, experiences: List[Dict], start_idx: int) -> List[Dict]:
        reassigned = []
        for i, exp in enumerate(experiences):
            updated = dict(exp)
            updated["experience_id"] = f"exp_{start_idx + i:06d}"
            reassigned.append(updated)
        return reassigned

    def _build_analysis_prompt(
        self,
        failed_trajectories: List[Dict],
        current_experiences: Dict,
        next_exp_idx: int,
    ) -> str:
        examples = []
        for i, traj in enumerate(failed_trajectories[:10]):
            examples.append(self._format_failure_example(i + 1, traj))

        existing = []
        for pool_name, experience in self._iter_existing_experiences(current_experiences):
            trigger = experience.get("trigger") or experience.get("when_to_apply") or experience.get("title") or ""
            recommended_action = experience.get("recommended_action") or experience.get("principle") or ""
            identifier = experience.get("experience_id") or "unknown-id"
            existing.append(
                f"[{pool_name}; {identifier}; {experience.get('task_type', 'unknown')}] "
                f"{trigger} -> {recommended_action}"
            )
        example_ids = ", ".join(
            f'"exp_{next_exp_idx + j:06d}"' for j in range(self.max_new_experiences_per_update)
        )

        return f"""You are updating an ALFWorld state-aware experience bank.

Analyze the contrasted rollout snippets below and propose NEW or revised experiences that help the agent act better in the same environment state.
These contrasts are selected when retrieved experience appears harmful: the rollout WITH experience failed, while the same-anchor rollout WITHOUT experience succeeded.

Requirements:
- Each experience must be state-aware, not a generic tip.
- Focus on what the successful no-experience rollout did, and what the retrieved experience may have caused the agent to do incorrectly.
- Prefer compact trigger language that can be matched at retrieval time.
- Use actionable wording tied to admissible actions and visible state cues.
- Return only JSON.

FAILED / CONTRASTED TRAJECTORIES:
{''.join(examples)}

EXISTING EXPERIENCES / EXPERIENCE TRIGGERS / ACTIONS:
{existing}

Generate 1-{self.max_new_experiences_per_update} new experiences.
Each item must contain:
- experience_id
- task_type
- trigger
- recommended_action
- avoid_action
- expected_effect
- state_cues (list[str])
- retrieval_text
- evidence {{
    "positive_gain_count": 0,
    "negative_gain_count": 0,
    "source_traj_uids": [],
    "last_updated_step": null
  }}

Use example ids: {example_ids}

Return ONLY a JSON array of experience objects.
"""

    # ------------------------------------------------------------------ #
    # Search-QA prompt (T064).  The upstream EDGE release only ships the  #
    # ALFWorld prompt above; this is a domain adaptation with the same    #
    # input selection (experience_hurt contrasts) and the same JSON       #
    # schema, used identically by every method.                           #
    # ------------------------------------------------------------------ #

    SEARCH_TASK_TYPES = ("direct_retrieval", "entity_attribute_lookup", "multi_hop_reasoning", "comparison")

    def _prompt_domain(self, current_experiences: Dict) -> str:
        forced = os.environ.get("EDGE_REFLECTOR_PROMPT", "auto").lower()
        if forced in ("search", "alfworld"):
            return forced
        env = ""
        if isinstance(current_experiences, dict):
            env = str((current_experiences.get("metadata") or {}).get("env") or "").lower()
        return "search" if env == "search" else "alfworld"

    @staticmethod
    def _extract_answer(text: str) -> str:
        m = re.findall(r"<answer>(.*?)</answer>", text or "", flags=re.S)
        return m[-1].strip() if m else ""

    def _experience_lookup(self, current_experiences: Dict) -> Dict[str, Dict]:
        table = {}
        for _, experience in self._iter_existing_experiences(current_experiences):
            identifier = str(experience.get("experience_id") or "")
            if identifier:
                table[identifier] = experience
        return table

    def _format_search_example(self, idx: int, traj: Dict, lookup: Dict[str, Dict]) -> str:
        if traj.get("fork_id"):
            pair = traj["paired_rollout"]
            return "\n".join([
                f"\nExample {idx}:", f"Question: {traj.get('task', '')}",
                f"Fork ID: {traj['fork_id']}",
                f"Fork depth: {traj.get('fork_depth')} (root=1, internal>1)",
                "Common earlier task experience: " + json.dumps(traj.get("common_task_experience", {}), ensure_ascii=False),
                "Search history before fork: " + json.dumps(traj.get("search_history", []), ensure_ascii=False),
                "New experiences at this fork: " + json.dumps(traj.get("added_experiences", []), ensure_ascii=False),
                f"LEFT action: {pair['without_experience'].get('action', '')}",
                f"RIGHT action: {pair['with_experience'].get('action', '')}",
                f"LEFT subtree success: {pair['without_experience'].get('successes')}/{pair['without_experience'].get('total')} = {pair['without_experience'].get('score')}",
                f"RIGHT subtree success: {pair['with_experience'].get('successes')}/{pair['with_experience'].get('total')} = {pair['with_experience'].get('score')}",
                f"Delta: {traj['delta']}; experience {traj['effect']}",
            ]) + "\n"
        pair = traj.get("paired_rollout", {}) or {}
        with_exp = pair.get("with_experience", {}) or {}
        without_exp = pair.get("without_experience", {}) or {}
        context = str(traj.get("anchor_obs", "") or "")
        if len(context) > 1500:
            context = "..." + context[-1500:]
        lines = [
            f"\nExample {idx}:",
            f"Question: {traj.get('task', '')}",
            f"Question type (heuristic): {traj.get('task_type', 'unknown')}",
            f"Context at this turn (tail, truncated): {context}",
            "Experiences that were injected into the failed rollout:",
        ]
        for eid in with_exp.get("experience_ids", []) or []:
            exp = lookup.get(str(eid), {})
            trig = exp.get("trigger") or exp.get("when_to_apply") or exp.get("title") or ""
            act = exp.get("recommended_action") or exp.get("principle") or ""
            lines.append(f"  [{eid}] {trig} -> {act}")
        lines.append(f"Rollout WITH experience (score={with_exp.get('score', '')}):\n  {with_exp.get('action', '')}")
        lines.append(f"Rollout WITHOUT experience (score={without_exp.get('score', '')}):\n  {without_exp.get('action', '')}")
        return "\n".join(lines) + "\n"

    def _build_search_analysis_prompt(
        self,
        failed_trajectories: List[Dict],
        current_experiences: Dict,
        next_exp_idx: int,
    ) -> str:
        lookup = self._experience_lookup(current_experiences)
        examples = [
            self._format_search_example(i + 1, traj, lookup)
            for i, traj in enumerate(failed_trajectories[:10])
        ]
        existing = []
        for pool_name, experience in self._iter_existing_experiences(current_experiences):
            trigger = experience.get("trigger") or experience.get("when_to_apply") or experience.get("title") or ""
            recommended_action = experience.get("recommended_action") or experience.get("principle") or ""
            identifier = experience.get("experience_id") or "unknown-id"
            existing.append(f"- [{identifier}; {experience.get('task_type', 'unknown')}] {trigger} -> {recommended_action}")
        example_ids = ", ".join(f'"exp_{next_exp_idx + j:06d}"' for j in range(self.max_new_experiences_per_update))
        types = ", ".join(self.SEARCH_TASK_TYPES)
        existing_text = "\n".join(existing) if existing else "(none)"

        return f"""You are maintaining an experience bank for a search agent that answers open-domain questions.
The agent works in turns: it reasons inside <think>...</think>, issues a search engine query inside <search>...</search>, reads the returned passages inside <information>...</information>, and finally answers inside <answer>...</answer>. A rollout is scored 1 if the final answer matches the reference and 0 otherwise.

Each example below is a contrast on the SAME question and the SAME turn: the rollout that received the listed experiences failed, while the rollout without any experience succeeded. So the injected experience either misled the agent or failed to prevent a mistake.

Your job: write reusable search strategies that would help on NEW questions of the same kind.

Rules (strict):
1. Generalize. Never copy any person, place, work, organization, date, number, or answer from the examples into the experience. Refer to them abstractly, e.g. "the person", "the film", "the second entity", "a year".
2. One experience = one strategy for one question pattern. Do not merge unrelated examples into one experience. It is fine to produce fewer experiences than allowed.
3. Explain the strategy through the agent's actions: how to phrase a search query, when to search again, how to check that a passage is about the right entity and relation, how to choose the answer span and its format.
4. Use the successful no-experience rollout to see what worked, and the injected experience to see what went wrong. If an injected experience is simply wrong for this pattern, write the corrected strategy and state in avoid_action what not to do.
5. Do not repeat an existing experience below; write only something new or a clearly better replacement.
6. task_type must be one of: {types}.

EXAMPLES:
{''.join(examples)}
EXISTING EXPERIENCES:
{existing_text}

Generate 1-{self.max_new_experiences_per_update} experiences. Each item must contain:
- experience_id (use: {example_ids})
- task_type (one of the allowed values)
- trigger: the question pattern or turn situation where this applies, in general terms
- recommended_action: what the agent should do (query phrasing / verification / answer selection)
- avoid_action: the mistake to avoid
- expected_effect: why this leads to a correct answer
- state_cues: list of short general cues visible in the question or passages (no example-specific names)
- retrieval_text: one or two general sentences used to match this experience to future questions
- source_examples: list of the Example numbers this experience is based on (integers)

Return ONLY a JSON array of experience objects.
"""

    def _postprocess_search_experiences(self, experiences: List[Dict], failed_trajectories: List[Dict]) -> List[Dict]:
        """Map source example numbers to real trajectory ids, reset LLM-provided counters,
        and drop experiences that copy an example's answer string verbatim."""
        answers = set()
        for traj in failed_trajectories[:10]:
            pair = traj.get("paired_rollout", {}) or {}
            for side in ("with_experience", "without_experience"):
                ans = self._extract_answer((pair.get(side, {}) or {}).get("action", ""))
                if len(ans) >= 3:
                    answers.add(ans.lower())
        kept = []
        self.last_answer_leak_dropped = 0
        for exp in experiences:
            text = " ".join(
                str(exp.get(k, "")) for k in ("trigger", "recommended_action", "avoid_action", "retrieval_text")
            ) + " " + " ".join(str(c) for c in (exp.get("state_cues") or []))
            low = text.lower()
            if any(a in low for a in answers):
                self.last_answer_leak_dropped += 1
                continue
            if exp.get("task_type") not in self.SEARCH_TASK_TYPES:
                exp["task_type"] = "multi_hop_reasoning" if "hop" in low else "direct_retrieval"
            sources = []
            for n in exp.get("source_examples", []) or []:
                try:
                    traj = failed_trajectories[int(n) - 1]
                except (ValueError, TypeError, IndexError):
                    continue
                if traj.get("source_leaf_ids"):
                    sources.extend(str(x) for x in traj["source_leaf_ids"])
                else:
                    for key in ("traj_uid", "no_exp_traj_uid"):
                        if traj.get(key):
                            sources.append(str(traj[key]))
            exp["evidence"] = {
                "positive_gain_count": 0,
                "negative_gain_count": 0,
                "source_traj_uids": sources,
                "last_updated_step": None,  # set by the trainer to the real global step
            }
            if any(x.get("fork_id") for x in failed_trajectories):
                source_depths = []
                for source in exp.get("source_examples", []) or []:
                    try:
                        source_depths.append(int(failed_trajectories[int(source)-1]["fork_depth"]))
                    except (ValueError, TypeError, KeyError, IndexError):
                        continue
                if exp.get("experience_type") not in ("task", "step"):
                    exp["experience_type"] = "task" if (source_depths[0] if source_depths else failed_trajectories[0].get("fork_depth", 2)) == 1 else "step"
            exp.pop("source_examples", None)
            kept.append(exp)
        if self.last_answer_leak_dropped:
            print(f"[ExperienceUpdater] dropped {self.last_answer_leak_dropped} experiences that copied an example answer", flush=True)
        return kept

    def _format_failure_example(self, idx: int, traj: Dict) -> str:
        task = traj.get("task", "")
        task_type = traj.get("task_type", "unknown")
        anchor = traj.get("anchor_obs", traj.get("anchor", ""))
        contrast_type = traj.get("contrast_type", "same_anchor_contrast")
        success = traj.get("trajectory_success", traj.get("episode_reward", None))
        admissible_actions = traj.get("admissible_actions", [])
        same_anchor_pair = traj.get("paired_rollout", None)

        lines = [
            f"\nExample {idx}:",
            f"Task: {task}",
            f"Task Type: {task_type}",
            f"Contrast Type: {contrast_type}",
            f"Anchor Observation: {anchor}",
            f"Admissible Actions: {admissible_actions}",
        ]
        if same_anchor_pair:
            lines.append("Paired Contrast:")
            with_exp = same_anchor_pair.get("with_experience", {}) or {}
            without_exp = same_anchor_pair.get("without_experience", {}) or {}
            lines.append(
                "  With experience: "
                f"action={with_exp.get('action', '')}; "
                f"score={with_exp.get('score', '')}; "
                f"reward={with_exp.get('reward', '')}; "
                f"experience_ids={with_exp.get('experience_ids', [])}; "
                f"query_hash={with_exp.get('query_hash', '')}"
            )
            lines.append(
                "  Without experience: "
                f"action={without_exp.get('action', '')}; "
                f"score={without_exp.get('score', '')}; "
                f"reward={without_exp.get('reward', '')}"
            )
        lines.append(f"Outcome: {success}")
        lines.append("Trajectory:")
        for step in traj.get("trajectory", [])[:3] + traj.get("trajectory", [])[-5:]:
            step_idx = step.get("step_idx", "unknown")
            observation = step.get("observation", "")
            action = step.get("action", "")
            lines.append(f"  Step {step_idx}: obs={observation[:300]} action={action[:300]}")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------ #
    # Parsing                                                             #
    # ------------------------------------------------------------------ #

    def _parse_experiences_response(self, response: str) -> List[Dict]:
        self.last_parse_succeeded = False
        try:
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            if json_start != -1 and json_end > json_start:
                items = json.loads(response[json_start:json_end])
                parsed = []
                for item in items:
                    if all(
                        key in item
                        for key in (
                            "experience_id",
                            "task_type",
                            "trigger",
                            "recommended_action",
                            "avoid_action",
                            "expected_effect",
                            "state_cues",
                            "retrieval_text",
                        )
                    ):
                        item.setdefault("evidence", {})
                        parsed.append(item)
                self.last_parse_succeeded = len(items) == 0 or len(parsed) > 0
                return parsed
        except json.JSONDecodeError as e:
            print(f"[ExperienceUpdater] JSON parse error: {e}")
        return []

    def get_update_summary(self) -> Dict:
        if not self.update_history:
            return {"total_updates": 0, "total_experiences_generated": 0}
        return {
            "total_updates": len(self.update_history),
            "total_experiences_generated": sum(
                h["num_experiences_generated"] for h in self.update_history
            ),
            "all_experience_ids": [
                sid for h in self.update_history for sid in h["experience_ids"]
            ],
        }

"""
LLM-based experience updater for ALFWorld experience banks.

The updater consumes contrasted rollout snippets and emits structured
state-aware experiences instead of generic experiences.
"""

import json
import os
import re
import time
import threading
import queue
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
from .experience_type_validation import CLASSIFICATION_GUIDANCE, validate_experience_types


class ExperienceUpdater:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_new_experiences_per_update: int = 5,
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
        if OpenAI is None:
            raise ImportError("The 'openai' package is required to initialize ExperienceUpdater client.")

        default_timeout = float(os.environ.get("EXPERIENCE_UPDATE_TIMEOUT_SECONDS", "330.0"))
        self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=default_timeout)
        self._audit_request_count = 0
        if os.environ.get("T063_REFLECTOR_AUDIT") == "1":
            self.client._client.event_hooks.setdefault("request", []).append(self._audit_request)
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.max_new_experiences_per_update = max_new_experiences_per_update
        self.token_limit_param = token_limit_param
        self.update_history = []
        # Safe defaults for intervals with no contrast examples.
        self.last_update_succeeded = False
        self.last_failure_reason = None
        self.last_classification_stats = {}
        self.last_classification_decisions = []

    def _audit_request(self, request):
        self._audit_request_count += 1

    def _completion_with_deadline(self, request_kwargs, total_seconds=None):
        if total_seconds is None:
            total_seconds = float(os.environ.get("EXPERIENCE_UPDATE_DEADLINE_SECONDS", "340.0"))
        if getattr(self, "_request_thread", None) is not None and self._request_thread.is_alive():
            raise TimeoutError("Previous reflector request is still ending")
        results = queue.Queue(maxsize=1)
        deadline = time.time() + total_seconds

        def request():
            try:
                client = self.client.with_options(
                    max_retries=0, timeout=max(0.1, total_seconds - 10),
                    default_headers={"X-Echo-Deadline": str(deadline)})
                results.put((True, client.chat.completions.create(**request_kwargs)))
            except Exception as error:
                results.put((False, error))

        self._request_thread = threading.Thread(target=request, daemon=True)
        self._request_thread.start()
        try:
            ok, result = results.get(timeout=total_seconds)
        except queue.Empty:
            raise TimeoutError("Reflector exceeded its total request deadline") from None
        if not ok:
            raise result
        return result

    def analyze_failures(
        self,
        failed_trajectories: List[Dict],
        current_experiences: Dict,
    ) -> List[Dict]:
        self.last_classification_stats = dict(
            submitted=0, accepted=0, rejected=0,
            ambiguous=0, failed=0, invalid_type_dropped=0,
        )
        self.last_classification_decisions = []
        if not failed_trajectories:
            return []

        self.last_update_succeeded = False
        self.last_failure_reason = None
        try:
            next_exp_idx = self._next_exp_index(current_experiences)
            domain = self._prompt_domain(current_experiences)
            if domain == "search":
                prompt = self._build_search_analysis_prompt(failed_trajectories, current_experiences, next_exp_idx)
                prompt += "\n" + CLASSIFICATION_GUIDANCE
            else:
                prompt = self._build_analysis_prompt(failed_trajectories, current_experiences, next_exp_idx)
            self.last_prompt_domain = domain

        except (ValueError, TypeError) as error:
            self.last_failure_reason = "prompt_" + type(error).__name__
            print(f"[ExperienceUpdater] prompt construction failed: {type(error).__name__}", flush=True)
            return []

        started = time.perf_counter()
        request_count_before = self._audit_request_count
        parsed_count = 0
        self.last_update_succeeded = False
        self.last_failure_reason = None
        request_succeeded = False
        try:
            request_kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                self.token_limit_param: self.max_completion_tokens,
            }
            response = self._completion_with_deadline(request_kwargs)
            request_succeeded = True
            raw_experiences = self._parse_experiences_response(response.choices[0].message.content)
            parsed_count = len(raw_experiences)
            self.last_update_succeeded = self.last_parse_succeeded
            if not self.last_parse_succeeded:
                self.last_failure_reason = "invalid_response_schema"
            if domain == "search":
                raw_experiences = self._postprocess_search_experiences(raw_experiences, failed_trajectories)
                self.last_classification_stats["invalid_type_dropped"] = int(
                    getattr(self, "last_invalid_fork_type_dropped", 0)
                )
                # By default, run 1 single LLM API call per reflection window.
                # Set ENABLE_EXPERIENCE_TYPE_VALIDATION=1 to enable the optional
                # second-pass type validator audit.
                enable_type_validator = os.environ.get(
                    "ENABLE_EXPERIENCE_TYPE_VALIDATION", "0"
                ).strip().lower() in ("1", "true", "yes")
                if any(t.get("fork_id") for t in failed_trajectories) and raw_experiences:
                    self.last_classification_stats["submitted"] = len(raw_experiences)
                    if enable_type_validator:
                        try:
                            raw_experiences, decisions = validate_experience_types(
                                raw_experiences, self._completion_with_deadline,
                                model=self.model,
                                token_limit_param=self.token_limit_param,
                                max_completion_tokens=self.max_completion_tokens,
                            )
                        except Exception as error:
                            self.last_classification_stats["failed"] = (
                                self.last_classification_stats["submitted"]
                            )
                            self.last_failure_reason = "classification_validation_failed"
                            self.last_update_succeeded = False
                            print("[ExperienceUpdater] classification validation failed: "
                                  + type(error).__name__, flush=True)
                            return []  # fail closed; leave the live bank unchanged
                        self.last_classification_decisions = decisions
                        self.last_classification_stats["accepted"] = len(raw_experiences)
                        self.last_classification_stats["rejected"] = sum(
                            not x["accepted"] for x in decisions
                        )
                        self.last_classification_stats["ambiguous"] = sum(
                            x["inferred_type"] == "ambiguous" for x in decisions
                        )
                        for decision in decisions:
                            if not decision["accepted"]:
                                print("[ExperienceUpdater] classification rejected: "
                                      + json.dumps(decision, ensure_ascii=False), flush=True)
                    else:
                        self.last_classification_decisions = []
                        self.last_classification_stats["accepted"] = len(raw_experiences)
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
            self.last_failure_reason = type(e).__name__
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
        # The bank retains IDs of deleted experiences in its watermark.
        # Do not propose a historical exp ID merely because it was evicted.
        marks = (current_experiences.get("metadata") or {}).get(
            "experience_id_high_watermarks", {}
        )
        if isinstance(marks, dict):
            max_idx = max(max_idx, int(marks.get("exp", 0) or 0))
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

    SEARCH_TASK_TYPES = (
        "general",
        "direct_retrieval",
        "entity_attribute_lookup",
        "multi_hop_reasoning",
        "comparison",
        "temporal",
        "numeric",
        "list_intersection",
        "source_identification",
        "location_chain",
        "answer_normalization",
    )

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

    @staticmethod
    def _resolve_experience_entries(entries: Any, lookup: Dict[str, Dict]) -> List[Dict]:
        if not entries:
            return []
        if isinstance(entries, dict):
            ids = entries.get("experience_ids") or []
            if ids:
                entries = ids
            elif entries.get("trigger") or entries.get("recommended_action") or entries.get("principle"):
                entries = [entries]
            else:
                return []
        resolved = []
        for item in entries if isinstance(entries, (list, tuple)) else [entries]:
            if isinstance(item, str):
                exp = lookup.get(item, {})
                if exp:
                    resolved.append({
                        "experience_id": item,
                        "trigger": exp.get("trigger") or exp.get("when_to_apply") or exp.get("title") or "",
                        "recommended_action": exp.get("recommended_action") or exp.get("principle") or "",
                        "avoid_action": exp.get("avoid_action") or "",
                    })
                else:
                    resolved.append({"experience_id": item})
            elif isinstance(item, dict):
                eid = str(item.get("experience_id") or "")
                exp = lookup.get(eid, {}) if eid else {}
                resolved.append({
                    "experience_id": eid or "unknown",
                    "trigger": item.get("trigger") or item.get("when_to_apply") or item.get("title") or exp.get("trigger") or exp.get("when_to_apply") or exp.get("title") or "",
                    "recommended_action": item.get("recommended_action") or item.get("principle") or exp.get("recommended_action") or exp.get("principle") or "",
                    "avoid_action": item.get("avoid_action") or exp.get("avoid_action") or "",
                })
        return resolved

    def _format_search_example(self, idx: int, traj: Dict, lookup: Dict[str, Dict]) -> str:
        if traj.get("fork_id"):
            pair = traj["paired_rollout"]
            fork_depth = int(traj.get("fork_depth") or 1)
            stage_desc = (
                "root=1 (initial query planning before any search; extract a task-level experience if a generalizable question pattern exists)"
                if fork_depth == 1
                else f"internal={fork_depth} (mid-search step after reading retrieved results; extract a step-level experience grounded in the observed search state)"
            )
            common_exps = self._resolve_experience_entries(traj.get("common_task_experience"), lookup)
            added_exps = self._resolve_experience_entries(
                traj.get("added_experiences") or pair.get("with_experience", {}).get("experience_ids"),
                lookup,
            )
            raw_history = traj.get("search_history") or []
            compact_history = []
            for h in raw_history:
                if isinstance(h, (list, tuple)) and len(h) >= 2:
                    compact_history.append({
                        "action": str(h[0] or "")[:300],
                        "observation": str(h[1] or "")[:300],
                    })
                elif isinstance(h, dict):
                    compact_history.append({
                        k: (str(v)[:300] if isinstance(v, str) else v)
                        for k, v in h.items()
                    })
                else:
                    compact_history.append(str(h)[:300])
            anchor_obs = str(traj.get("anchor_obs") or "")[:800]
            # Only show post-fork turns (depth >= fork_depth) in each branch to avoid
            # repeating the identical pre-fork search history three times.
            success_turns = [
                {"depth": t.get("depth"), "action": (t.get("action") or "")[:400],
                 "observation": (t.get("observation") or "")[:500]}
                for t in (traj.get("success_turns") or [])
                if int(t.get("depth") or fork_depth) >= fork_depth
            ]
            failed_turns = [
                {"depth": t.get("depth"), "action": (t.get("action") or "")[:400],
                 "observation": (t.get("observation") or "")[:500]}
                for t in (traj.get("failed_turns") or [])
                if int(t.get("depth") or fork_depth) >= fork_depth
            ]
            left_action = str(pair["without_experience"].get("action", "") or "")[:3000]
            right_action = str(pair["with_experience"].get("action", "") or "")[:3000]
            return "\n".join([
                f"\nExample {idx}:",
                f"Question: {traj.get('task', '')}",
                f"Fork ID: {traj['fork_id']}",
                f"Fork depth: {stage_desc}",
                "Common earlier task experience: " + json.dumps(common_exps, ensure_ascii=False),
                "Search history before fork: " + json.dumps(compact_history, ensure_ascii=False),
                f"Observation at fork: {anchor_obs}",
                "Successful trajectory: " + json.dumps(success_turns, ensure_ascii=False),
                "Failed trajectory: " + json.dumps(failed_turns, ensure_ascii=False),
                "New experiences injected into RIGHT at this fork (harmful/misleading in this state): " + json.dumps(added_exps, ensure_ascii=False),
                f"LEFT action (without new experience, succeeded): {left_action}",
                f"RIGHT action (with new experience, failed): {right_action}",
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
            for i, traj in enumerate(failed_trajectories[:12])
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

        def _render(ex_list):
            return f"""You are maintaining an experience bank for a search agent that answers open-domain questions.
The agent works in turns: it reasons inside <think>...</think>, issues a search engine query inside <search>...</search>, reads the returned passages inside <information>...</information>, and finally answers inside <answer>...</answer>. A rollout is scored 1 if the final answer matches the reference and 0 otherwise.

Each example below is a controlled fork at one identical search state on the SAME question: LEFT omitted this turn's newly retrieved experience and succeeded, while RIGHT received the listed experience and failed (Delta < 0). Thus the injected experience either misled the agent (e.g., over-searching, wrong disambiguation, or premature/wrong query focus) or failed to provide the strategy needed at that state.

Your job: write reusable, state-aware search strategies that would help on NEW questions in the same situation.

Rules (strict):
1. Generalize. Never copy any person, place, work, organization, date, number, or answer from the examples into the experience. Refer to them abstractly, e.g. "the person", "the film", "the second entity", "a year".
2. One experience = one strategy for one question pattern or search state. Do not merge unrelated examples into one experience. Quality over quantity: it is completely fine to produce fewer experiences than the maximum allowed (or 0 if no clear generalizable lesson exists).
3. Explain the strategy through the agent's concrete actions: how to phrase a search query, when to refine a query after noisy/partial results, how to verify that a passage matches the exact entity and relation, and when to stop searching and output <answer> directly once sufficient evidence is present.
4. Compare the successful LEFT trajectory against the failed RIGHT trajectory and the experience injected at the fork. Write every field as direct advice to the agent: never mention forks, LEFT/RIGHT, scores, subtree success, examples, or how this experience was derived.
5. STRICT ANTI-DUPLICATION: Never restate or lightly reword an existing rule. However, for mid-search (step) examples, do NOT skip a concrete, narrower failure pattern (such as a truncated answer-bearing clause, a biography date mistaken for an institution's founding year, or wrong geographic granularity) merely because a broad generic rule like "refine query" or "disambiguate" already exists—extract the specific, actionable step refinement.
6. task_type must be one of: {types}.
7. For each object include experience_type ("task" or "step"):
   - "task": applicability (trigger, state_cues, and retrieval_text) depends ONLY on the initial question structure before any search is executed.
   - "step": applicability (trigger, state_cues, and retrieval_text) depends on the mid-search state after reading retrieved snippets or search history.

EXAMPLES:
{''.join(ex_list)}
EXISTING EXPERIENCES:
{existing_text}

Generate 0-{self.max_new_experiences_per_update} experiences. Each item must contain:
- experience_id (use: {example_ids})
- experience_type ("task" or "step")
- task_type (one of the allowed values)
- trigger: the question pattern (for task) or mid-search observation state (for step) where this applies, in general terms
- recommended_action: what the agent should do (query phrasing / verification / stopping search to answer)
- avoid_action: the mistake to avoid
- expected_effect: why this leads to a correct answer
- state_cues: list of short general cues visible in the question (for task) or in the retrieved passages/history (for step)
- retrieval_text: one or two purely declarative state-description sentences matching retrieval queries 鈥?describe ONLY the question structure (for task, e.g., "A multi-hop question asking for...") or observable snippet status (for step, e.g., "Observation status: multiple candidate films share the same title..."). Do NOT start with imperative verbs like "Use this", "Apply this", or "Search for"
- source_examples: list of the Example numbers this experience is based on (integers)

Return ONLY a JSON array of experience objects.
"""

        prompt = _render(examples)
        if len(prompt) > 180000:
            raise ValueError("Reflector prompt exceeds 180000 characters")
        self.last_prompt_chars = len(prompt)
        return prompt

    def _postprocess_search_experiences(self, experiences: List[Dict], failed_trajectories: List[Dict]) -> List[Dict]:
        """Map source example numbers to real trajectory ids, reset LLM-provided counters,
        and drop experiences that copy an example's answer string verbatim."""
        answers = set()
        for traj in failed_trajectories[:12]:
            pair = traj.get("paired_rollout", {}) or {}
            for side in ("with_experience", "without_experience"):
                ans = self._extract_answer((pair.get(side, {}) or {}).get("action", ""))
                if len(ans) >= 3:
                    answers.add(ans.lower())
            for key in ("failed_turns", "success_turns"):
                for t in traj.get(key) or []:
                    ans = self._extract_answer(t.get("action", ""))
                    if len(ans) >= 3:
                        answers.add(ans.lower())
        kept = []
        self.last_answer_leak_dropped = 0
        is_v3 = any(t.get("reflector_version") == "v3" for t in failed_trajectories)
        # Fork-selected T72 extraction follows EDGE: one model call, then
        # exact-fingerprint bank upsert. No second answer-string novelty gate.
        is_edge_fork = any(t.get("fork_id") for t in failed_trajectories) and not is_v3
        self.last_invalid_v3_dropped = 0
        self.last_invalid_fork_type_dropped = 0
        for exp in experiences:
            # For T73 Fork + EDGE, invalid/missing labels are errors rather
            # than an invitation to guess from the source Fork depth.
            if is_edge_fork and exp.get("experience_type") not in ("task", "step"):
                self.last_invalid_fork_type_dropped += 1
                continue
            if is_v3:
                # LLMs may use the human-facing T/S abbreviations even when
                # asked for task/step. This is a schema alias, NOT inference
                # from fork depth or from the observed outcome.
                kind = str(exp.get("experience_type") or "").strip().lower()
                exp["experience_type"] = {
                    "t": "task", "s": "step",
                    "task-level": "task", "step-level": "step",
                    "task_experience": "task", "step_experience": "step",
                }.get(kind, kind)
                sources = exp.get("source_examples")
                valid_sources = isinstance(sources, list) and any(
                    str(x).isdigit() and 1 <= int(x) <= len(failed_trajectories)
                    for x in sources
                )
                if exp.get("experience_type") not in ("task", "step") or not valid_sources:
                    # v3 MUST classify by actual strategy and cite an example;
                    # never silently infer T/S from fork depth.
                    self.last_invalid_v3_dropped += 1
                    continue
            text = " ".join(
                str(exp.get(k, "")) for k in ("trigger", "recommended_action", "avoid_action", "retrieval_text")
            ) + " " + " ".join(str(c) for c in (exp.get("state_cues") or []))
            low = text.lower()
            if not is_edge_fork and any(a in low for a in answers):
                self.last_answer_leak_dropped += 1
                continue
            if exp.get("task_type") not in self.SEARCH_TASK_TYPES:
                exp["task_type"] = "multi_hop_reasoning" if "hop" in low else "direct_retrieval"
            if exp.get("retrieval_text"):
                rt = str(exp["retrieval_text"]).strip()
                rt = re.sub(
                    r"^(?:use|apply)\s+this\s+(?:strategy|rule|experience|guidance)\s+(?:when|for|after|if)\s+",
                    "",
                    rt,
                    flags=re.I,
                ).strip()
                if rt:
                    exp["retrieval_text"] = rt[0].upper() + rt[1:]
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
            if any(x.get("mode") == "trajectory_pair" for x in failed_trajectories):
                if exp.get("experience_type") not in ("task", "step"):
                    depth = 2
                    for source in exp.get("source_examples", []) or []:
                        try:
                            depth = int(failed_trajectories[int(source) - 1].get("divergence_depth", 2))
                            break
                        except (ValueError, TypeError, IndexError):
                            continue
                    exp["experience_type"] = "task" if depth == 1 else "step"
            elif any(x.get("fork_id") for x in failed_trajectories):
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
                payload = response[json_start:json_end]
                self.last_json_repaired = False
                try:
                    items = json.loads(payload)
                except json.JSONDecodeError:
                    # Model sometimes writes a template like
                    # \`"[entity]" relation\` inside a JSON string without
                    # escaping its internal quotes. Correct only this
                    # unambiguous, tightly scoped formatting error.
                    fixed = re.sub(
                        r'\`"([^"\n]{1,220})"',
                        lambda m: "`" + chr(92) + '"' + m.group(1)
                                  + chr(92) + '"',
                        payload,
                    )
                    if fixed == payload:
                        raise
                    items = json.loads(fixed)
                    self.last_json_repaired = True
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

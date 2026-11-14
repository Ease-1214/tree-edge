"""T069 prompt and bounded-request regressions."""
from types import SimpleNamespace
import time
import pytest
from agent_system.memory.experience_updater import ExperienceUpdater
from agent_system.tree.core import trajectory_reflector_examples
from test_tree_method import example

def updater():
    result=object.__new__(ExperienceUpdater)
    result.max_new_experiences_per_update=3
    return result

def test_trajectory_prompt_allows_uncertainty_and_no_rewording():
    tree=example()
    experiences=trajectory_reflector_examples(tree,k=10,seed=5)
    prompt=updater()._build_trajectory_pair_prompt(experiences,{},1)
    # v3 is intentionally unrestricted by the former 60k-character cap.
    assert "the final outcome alone never proves" in prompt
    assert "0 is okay" in prompt
    assert "EXPERIENCE TYPE = WHEN IT IS RETRIEVED AND USED" in prompt
    assert "treat it as unknown, not absent" in prompt
    assert "COMPLETE EXAMPLES" in prompt

def test_experience_ids_avoid_repeated_body_and_budget_scales_observations():
    turn=dict(depth=1,task_experience_ids=["task_1"],step_experience_ids=[],
        experience_ids=["task_1"],action="<search>entity</search>",observation="evidence "*500)
    item=dict(task="a question",failed_turns=[turn]*4,success_turns=[turn]*4,
        divergence_depth=1,leaf_success_rate=.1,n_leaves=10)
    writer=updater()
    prompt=writer._build_trajectory_pair_prompt([item]*10,
        {"task_experiences":[dict(experience_id="task_1",trigger="condition",recommended_action="unique body")]},1)
    assert len(prompt)<=60000
    assert prompt.count("unique body")==1
    assert "[TRUNCATED]" in prompt
    assert 128<=writer.last_observation_char_limit<=1200

def test_deadline_returns_without_waiting_for_slow_network():
    class FakeClient:
        def with_options(self,**kwargs):
            assert kwargs["max_retries"]==0
            assert "X-Echo-Deadline" in kwargs["default_headers"]
            self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
            return self
        def create(self,**kwargs):
            time.sleep(.15)
            return "done"
    writer=updater();writer.client=FakeClient()
    started=time.monotonic()
    with pytest.raises(TimeoutError):
        writer._completion_with_deadline({},total_seconds=.02)
    assert time.monotonic()-started<.12
    with pytest.raises(TimeoutError):
        writer._completion_with_deadline({},total_seconds=.02)

def test_prompt_failure_is_reported_without_network_call():
    writer=updater()
    writer._prompt_domain=lambda _: "search"
    result=writer.analyze_failures([dict(mode="trajectory_pair",task="x"*70000)],{})
    assert result==[]
    assert not writer.last_update_succeeded
    assert writer.last_failure_reason=="prompt_ValueError"


def test_removed_injected_experience_still_has_catalog_definition():
    turn=dict(depth=2,task_experience_ids=[],step_experience_ids=["removed_1"],
        experience_ids=["removed_1"],step_experiences=["specific evidence condition and action"],
        action="<answer>value</answer>",observation="")
    prompt=updater()._build_trajectory_pair_prompt([dict(task="question",failed_turns=[turn])],{},1)
    assert "removed_1; step; no longer in current bank" in prompt
    assert "specific evidence condition and action" in prompt

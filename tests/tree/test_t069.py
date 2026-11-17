"""T069/T75 prompt and bounded-request regressions."""
from types import SimpleNamespace
import time
import pytest
from agent_system.memory.experience_updater import ExperienceUpdater
from agent_system.tree.core import trajectory_reflector_examples
from agent_system.tree.edge_fork_adapter import edge_search_examples_from_forks
from test_tree_method import example

def updater():
    result=object.__new__(ExperienceUpdater)
    result.max_new_experiences_per_update=5
    result.last_update_succeeded=False
    result.last_failure_reason=None
    result.last_classification_stats={}
    result.last_classification_decisions=[]
    return result

def test_search_prompt_includes_anti_duplication_and_resolved_experience():
    tree=example((1,),(0,))
    tree.bank_snapshot={"task_experiences":[{"experience_id":"e","trigger":"when entity unknown","recommended_action":"search entity"}]}
    forks=trajectory_reflector_examples(tree,k=10,seed=5)
    edge_examples=edge_search_examples_from_forks(forks)
    prompt=updater()._build_search_analysis_prompt(edge_examples,tree.bank_snapshot,1)
    assert "STRICT ANTI-DUPLICATION" in prompt
    assert "Generate 0-5 experiences" in prompt
    assert "when entity unknown" in prompt

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
    result=writer.analyze_failures([dict(task="x"*190000)],{})
    assert result==[]
    assert not writer.last_update_succeeded
    assert writer.last_failure_reason=="prompt_ValueError"

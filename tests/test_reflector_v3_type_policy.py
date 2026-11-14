"""EDGE-style usage-stage T/S definitions without generation quotas or hard heuristics."""
import json
from types import SimpleNamespace

import pytest

from agent_system.memory.reflector_prompt_v3 import build_reflector_v3_prompt
from agent_system.memory.experience_updater import ExperienceUpdater


@pytest.fixture(autouse=True)
def no_proxy(monkeypatch):
    for key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy",
                "HTTPS_PROXY", "https_proxy"):
        monkeypatch.delenv(key, raising=False)


def updater():
    return ExperienceUpdater(
        api_key="unit-test", base_url="http://127.0.0.1:9/v1",
        model="fake", max_new_experiences_per_update=5,
    )


def proposal(kind, trigger):
    return dict(
        experience_id="exp_000001", experience_type=kind,
        task_type="multi_hop_reasoning", trigger=trigger,
        recommended_action="Check the correct entity and search its attribute",
        avoid_action="Do not use the wrong entity",
        expected_effect="Correct answer",
        state_cues=["identifying relation"], retrieval_text=trigger,
        source_examples=[1],
    )


EXAMPLE=dict(
    mode="trajectory_pair", reflector_version="v3", fork_depth=3,
    task="Which person won the contest?", source_leaf_ids=["left","right"],
    failed_turns=[], success_turns=[],
)


def test_prompt_neither_prioritizes_nor_forces_t():
    prompt=build_reflector_v3_prompt([EXAMPLE],{},1)
    assert "EXPERIENCE TYPE = WHEN IT IS RETRIEVED AND USED" in prompt
    assert "do not require either type" in prompt
    assert "Do not prefer task over step" in prompt
    assert "first turn" in prompt and "latest" in prompt
    assert "0 is okay" in prompt


@pytest.mark.parametrize("kind", ["task","step"])
def test_type_is_preserved_without_fork_depth_inference(kind):
    result=updater()._postprocess_search_experiences(
        [proposal(kind, "a detectable condition")], [EXAMPLE]
    )
    assert len(result)==1
    assert result[0]["experience_type"]==kind


def test_no_third_api_model_review(monkeypatch):
    worker=updater()
    calls=[]
    def fake_completion(kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps([proposal("step","Latest observation gives an ambiguity")])
        ))])
    monkeypatch.setattr(worker,"_completion_with_deadline",fake_completion)
    monkeypatch.setattr(worker,"_build_trajectory_pair_prompt",
                        lambda *args,**kwargs:"a prompt")
    result=worker.analyze_failures([EXAMPLE],{"task_experiences":[],"step_experiences":[]})
    assert len(calls)==1
    assert len(result)==1 and result[0]["experience_type"]=="step"

"""Reconstruct exactly the historic 10 Fork examples for Step 5/10/15.

Read-only; use the saved tree objects and previous active bank snapshots.
"""
import json
from pathlib import Path
import torch
from agent_system.tree.core import trajectory_reflector_examples
from agent_system.tree.edge_fork_adapter import edge_search_examples_from_forks
from agent_system.memory.experience_updater import ExperienceUpdater

RUN = Path('/home/zehao/echo-collab/T073/qwen25-3b-r64a128-a40x2-full11015-20261115')
INITIAL = Path('/home/zehao/echo-collab/T073/t71replica-r64-eightgpu-eval50-source/memory_data/search/state_aware_experiences_v2.json')

def original_bank(step):
    if step == 5:
        from agent_system.memory.experience_memory import ExperienceMemory
        return ExperienceMemory(str(INITIAL), retrieval_mode='template').experiences
    return json.loads((RUN/'checkpoints'/f'experience_bank_step{step-5}.json').read_text())

def main():
    out=Path(__file__).resolve().parents[1]/'reports/ab_prompts'
    out.mkdir(parents=True, exist_ok=True)
    updater=ExperienceUpdater()
    for step in (5,10,15):
        path=RUN/'tree-private'/f'tree-step{step}.pt'
        tree=torch.load(path,map_location='cpu',weights_only=False)
        forks=trajectory_reflector_examples(tree,k=100000,seed=step,zero_success_quota=0,sampling='random')[:10]
        saved=json.loads((RUN/'checkpoints'/f'tree_management_step{step}.json').read_text())['reflector_source_leaf_ids']
        actual=[x['source_leaf_ids'] for x in forks]
        if actual!=saved:
            print('REPLAY_INPUT_MISMATCH',step,'expected',saved,'actual',actual, flush=True)
            raise SystemExit(2)
        examples=edge_search_examples_from_forks(forks)
        bank=original_bank(step)
        nextid=updater._next_exp_index(bank)
        common=updater._build_search_analysis_prompt(examples,bank,nextid)
        before=common.replace(
            "Each example below is a contrast on the SAME question and the SAME turn: the rollout that received the listed experiences failed, while the rollout without any experience succeeded. So the injected experience either misled the agent or failed to prevent a mistake.",
            "Each example is a fork at one identical search state. LEFT omits only this turn's new experiences; RIGHT adds them. Both sides may retain earlier task experience. The actions shown are this turn's real actions; scores are terminal subtree success fractions. Delta=RIGHT-LEFT can show helped, harmed, or neutral evidence. Distill a reusable strategy from either sign without treating a noisy fork as certain causality."
        )
        before += "\nFor each object include experience_type: task or step. task is a reusable multi-turn solving strategy, usually from a root fork. step is an action or warning for a concrete state, usually from an internal fork. You need not produce both types.\n"
        before=before.replace(
            "Use the successful no-experience rollout to see what worked, and the injected experience to see what went wrong.",
            "Compare the two actions and subtree scores to identify a useful strategy or a harmful application of experience. Write every field as direct advice to the agent: never mention forks, LEFT/RIGHT, scores, subtree success, examples, or why this experience was derived."
        )
        # The current code enhances only the classification definitions; no input changes.
        from agent_system.memory.experience_type_validation import CLASSIFICATION_GUIDANCE
        after=before.replace(
            "task is a reusable multi-turn solving strategy, usually from a root fork. step is an action or warning for a concrete state, usually from an internal fork.",
            "Classify strictly by what information the trigger needs; source Fork depth is not a criterion."
        ) + "\n" + CLASSIFICATION_GUIDANCE
        (out/f'step{step}-baseline-prompt.txt').write_text(before)
        (out/f'step{step}-revised-prompt.txt').write_text(after)
        print('EXACT_INPUT_RECONSTRUCTED',step,'forks',len(forks),'candidates',len(trajectory_reflector_examples(tree,k=100000,seed=step,zero_success_quota=0,sampling='random')),'orig_chars',len(before),'revision_chars',len(after),'bank_count',sum(len(bank.get(p,[])) for p in ('task_experiences','step_experiences')),flush=True)

if __name__ == '__main__':
    main()

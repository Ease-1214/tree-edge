from agent_system.tree.core import TreeRollout,ForkRecord
from agent_system.tree.training import tree_exposure_counts,apply_exposure_counts


def test_three_injections_same_experience():
    tree=TreeRollout()
    tree.forks={"one":ForkRecord("one","q",1,"L","R",("e","e"),(0,0)),"two":ForkRecord("two","q",2,"LL","LR",("e",),(0,))}
    counts=tree_exposure_counts(tree)
    bank={"task_experiences":[{"experience_id":"e","retrieval_count":7}],"step_experiences":[]}
    apply_exposure_counts(bank,counts,20)
    assert counts["e"]==3 and bank["task_experiences"][0]["retrieval_count"]==10
    assert bank["task_experiences"][0]["last_retrieval_step"]==20

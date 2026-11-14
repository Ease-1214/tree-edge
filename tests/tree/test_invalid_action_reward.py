from test_tree_method import example
from agent_system.tree.core import leaf_advantages

def test_invalid_actions_affect_advantage_without_changing_em_or_gate():
    tree=example((0,),(1,))
    tree.invalid_action_penalty_coef=.01
    tree.nodes["R"].valid=False
    assert tree.leaves[1].reward==1
    assert tree.evaluate_forks()[0]["gate"]
    rows=tree.expanded_rows()
    assert tree.audits[0]["delta"]==1
    assert tree.audits[0]["weight"]==.5
    assert rows[0]["advantage"]<0 and rows[1]["advantage"]>0

def test_all_failed_group_learns_from_action_validity():
    tree=example((0,),(0,))
    tree.invalid_action_penalty_coef=.01
    tree.nodes["R"].valid=False
    scores=tree.training_advantages()
    assert scores["L0"]>0 and scores["R0"]<0
    assert not tree.evaluate_forks()[0]["gate"]
    assert tree.audits[0]["delta"]==0

def test_disabling_penalty_preserves_leaf_em_advantages():
    tree=example((0,),(0,))
    tree.nodes["R"].valid=False
    assert tree.training_advantages()==leaf_advantages(tree.leaves)

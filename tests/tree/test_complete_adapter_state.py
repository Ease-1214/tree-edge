import random
import numpy as np
import torch
from verl.utils.checkpoint.adapter_state import save_adapter_training_state,load_adapter_training_state


def test_three_steps_match_interrupted_lora(tmp_path):
    def setup():
        torch.manual_seed(7);np.random.seed(7);random.seed(7)
        a=torch.nn.Parameter(torch.randn(4,2));b=torch.nn.Parameter(torch.randn(2,3))
        opt=torch.optim.AdamW([a,b],lr=.001);sched=torch.optim.lr_scheduler.StepLR(opt,step_size=1,gamma=.9)
        return a,b,opt,sched
    def step(a,b,opt,sched):
        opt.zero_grad();x=torch.randn(2,4)*(np.random.rand()+random.random());loss=(x@a@b).square().mean();loss.backward();opt.step();sched.step()
    a,b,opt,sched=setup()
    for _ in range(3):step(a,b,opt,sched)
    expected=[a.detach().clone(),b.detach().clone()]
    a,b,opt,sched=setup();step(a,b,opt,sched)
    weights=[a.detach().clone(),b.detach().clone()]
    save_adapter_training_state(tmp_path/"state.pt",opt,sched)
    a,b,opt,sched=setup()
    with torch.no_grad():a.copy_(weights[0]);b.copy_(weights[1])
    assert load_adapter_training_state(tmp_path/"state.pt",opt,sched)
    for _ in range(2):step(a,b,opt,sched)
    torch.testing.assert_close(a,expected[0],rtol=0,atol=0);torch.testing.assert_close(b,expected[1],rtol=0,atol=0)

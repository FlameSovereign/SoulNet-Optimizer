import torch
from torch.optim.optimizer import Optimizer

class CollapseGrammarOptimizer_vGH1(Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if 'GH_trace' not in state:
                    state['GH_trace'] = torch.zeros_like(p.data)

                gh_trace = state['GH_trace']
                gh_trace.mul_(0.95).add_(0.05 * grad)

                update = grad - gh_trace
                p.data.add_(-group['lr'], update)
        return loss


import torch


def create():
    import timm

    model = timm.create_model('efficientnetv2_rw_m', True,
                              num_classes=15501, drop_rate=0.3, drop_path_rate=0.6)

    model.cuda()

    return model


class EMA:
    def __init__(self, model, decay=0.9999):
        super().__init__()
        import copy
        self.decay = decay
        self.model = copy.deepcopy(model)

        self.model.eval()

    def update(self, model):
        with torch.no_grad():
            e_std = self.model.state_dict().values()
            m_std = model.module.state_dict().values()
            for e, m in zip(e_std, m_std):
                e.copy_(self.decay * e + (1. - self.decay) * m)


class StepLR:
    def __init__(self, optimizer):
        self.optimizer = optimizer

        self.lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]

        self.decay_rate = 0.97
        self.decay_epochs = 2.4
        self.warmup_epochs = 3.0
        self.warmup_lr_init = 1e-6

        self.warmup_steps = [(lr - self.warmup_lr_init) / self.warmup_epochs for lr in self.lrs]
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.warmup_lr_init

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lrs = [self.warmup_lr_init + epoch * lr for lr in self.warmup_steps]
        else:
            lrs = [lr * (self.decay_rate ** (epoch // self.decay_epochs)) for lr in self.lrs]

        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr


class RMSprop(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, alpha=0.9, eps=1e-3, weight_decay=0, momentum=0.9,
                 centered=False, decoupled_decay=False, lr_in_momentum=True):

        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum,
                        centered=centered, decoupled_decay=decoupled_decay, lr_in_momentum=lr_in_momentum)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for param_group in self.param_groups:
            param_group.setdefault('momentum', 0)
            param_group.setdefault('centered', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for param_group in self.param_groups:
            for param in param_group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Optimizer does not support sparse gradients')
                state = self.state[param]
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.ones_like(param.data)
                    if param_group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(param.data)
                    if param_group['centered']:
                        state['grad_avg'] = torch.zeros_like(param.data)

                square_avg = state['square_avg']
                one_minus_alpha = 1. - param_group['alpha']

                state['step'] += 1

                if param_group['weight_decay'] != 0:
                    if 'decoupled_decay' in param_group and param_group['decoupled_decay']:
                        param.data.add_(param.data, alpha=-param_group['weight_decay'])
                    else:
                        grad = grad.add(param.data, alpha=param_group['weight_decay'])

                square_avg.add_(grad.pow(2) - square_avg, alpha=one_minus_alpha)

                if param_group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.add_(grad - grad_avg, alpha=one_minus_alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).add(param_group['eps']).sqrt_()
                else:
                    avg = square_avg.add(param_group['eps']).sqrt_()

                if param_group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    if 'lr_in_momentum' in param_group and param_group['lr_in_momentum']:
                        buf.mul_(param_group['momentum']).addcdiv_(grad, avg, value=param_group['lr'])
                        param.data.add_(-buf)
                    else:
                        buf.mul_(param_group['momentum']).addcdiv_(grad, avg)
                        param.data.add_(-param_group['lr'], buf)
                else:
                    param.data.addcdiv_(grad, avg, value=-param_group['lr'])

        return loss


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, output, target):
        prob = self.softmax(output)
        loss = -(prob.gather(dim=-1, index=target.unsqueeze(1))).squeeze(1)

        return ((1. - self.epsilon) * loss - self.epsilon * prob.mean(dim=-1)).mean()

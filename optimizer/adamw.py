import math
import torch
from torch.optim.optimizer import Optimizer


class AdamW(Optimizer):
    """Implements Adam algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_

    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        #super(AdamW, self).__init__(params, defaults)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    decayed_weights = torch.mul(p.data, group['weight_decay'])
                    p.data.addcdiv_(-step_size, exp_avg, denom)
                    p.data.sub_(decayed_weights)
                else:
                    p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss

from torch.optim import Optimizer
import math
import torch


class CosineLRWithRestarts():
    """Decays learning rate with cosine annealing, normalizes weight decay
    hyperparameter value, implements restarts.
    https://arxiv.org/abs/1711.05101
    https://techburst.io/improving-the-way-we-work-with-learning-rate-5e99554f163b
    https://hackernoon.com/how-to-train-your-neural-network-cb5a36e91072
    https://m-alcu.github.io/blog/2018/01/30/changing-learning-rate/
    https://medium.com/@nachiket.tanksale/finding-good-learning-rate-and-the-one-cycle-policy-7159fe1db5d6

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        batch_size: minibatch size
        epoch_size: training samples per epoch
        restart_period: epoch count in the first restart period
        t_mult: multiplication factor by which the next restart period will extend/shrink


    Example:
        >>> scheduler = CosineLRWithRestarts(optimizer, 32, 1024, restart_period=5, t_mult=1.2)
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>         ...
        >>>         optimizer.zero_grad()
        >>>         loss.backward()
        >>>         optimizer.step()
        >>>         scheduler.batch_step()
        >>>     validate(...)
    """

    def __init__(self, optimizer, batch_size, epoch_size, restart_period=100,
                 t_mult=2, last_epoch=-1, eta_threshold=1000, verbose=False):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an"
                                   " optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'],
                                 optimizer.param_groups))

        self.last_epoch = last_epoch
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.eta_threshold = eta_threshold
        self.t_mult = t_mult
        self.verbose = verbose
        self.base_weight_decays = list(map(lambda group: group['weight_decay'],
                                           optimizer.param_groups))
        self.restart_period = restart_period
        self.restarts = 0
        self.t_epoch = -1

    def _schedule_eta(self):
        """
        Threshold value could be adjusted to shrink eta_min and eta_max values.
        """
        eta_min = 0
        eta_max = 1
        if self.restarts <= self.eta_threshold:
            return eta_min, eta_max
        else:
            d = self.restarts - self.eta_threshold
            k = d * 0.09
            return (eta_min + k, eta_max - k)

    def get_lr(self, t_cur):
        eta_min, eta_max = self._schedule_eta()

        eta_t = (eta_min + 0.5 * (eta_max - eta_min)
                 * (1. + math.cos(math.pi *
                                  (t_cur / self.restart_period))))

        weight_decay_norm_multi = math.sqrt(self.batch_size /
                                            (self.epoch_size *
                                             self.restart_period))
        lrs = [base_lr * eta_t for base_lr in self.base_lrs]
        weight_decays = [base_weight_decay * eta_t * weight_decay_norm_multi
                         for base_weight_decay in self.base_weight_decays]

        if self.t_epoch % self.restart_period < self.t_epoch:
            if self.verbose:
                print("Restart at epoch {}".format(self.last_epoch))
            self.restart_period *= self.t_mult
            self.restarts += 1
            self.t_epoch = 0

        return zip(lrs, weight_decays)

    def _set_batch_size(self):
        d, r = divmod(self.epoch_size, self.batch_size)
        batches_in_epoch = d + 2 if r > 0 else d + 1
        self.batch_increment = iter(torch.linspace(0, 1, batches_in_epoch))

    def step(self):
        self.last_epoch += 1
        self.t_epoch += 1
        self._set_batch_size()
        self.batch_step()

    def batch_step(self):
        t_cur = self.t_epoch + next(self.batch_increment)
        for param_group, (lr, weight_decay) in zip(self.optimizer.param_groups,
                                                   self.get_lr(t_cur)):
            param_group['lr'] = lr
            param_group['weight_decay'] = weight_decay
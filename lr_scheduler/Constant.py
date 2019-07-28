from functools import partial

from torch._six import inf
from torch.optim import Optimizer


class Constant(object):
    def __init__(self, optimizer, eval_mode='min', threshold=1e-4, threshold_mode='abs',
                 last_batch_iteration=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        self.step(metrics=0, epoch=None, batch_iteration=last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

        self.eval_mode = eval_mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.last_epoch = -1
        self._init_is_better(eval_mode=eval_mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def _cmp(self, eval_mode, threshold_mode, threshold, a, best):
        if eval_mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            return a < best * rel_epsilon

        elif eval_mode == 'min' and threshold_mode == 'abs':
            return a < best - threshold

        elif eval_mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + threshold

    def step_epoch(self, metrics, epoch):
        text = ""

        current = metrics
        if epoch is None:
            epoch = self.last_epoch + 1
        assert epoch > self.last_epoch
        epoch_diff = int(epoch - self.last_epoch)
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            text = text + """
        Its Highest Score: {} --> {} (+{})
        NumBadEpoch: {} <= {} --> 0
        Coef: {}
        Times Reduce = {}
        """.format(self.best, current, current - self.best, self.num_bad_epochs, 0, 0, 0)
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += epoch_diff
            text = text + """
        Current: {}, Best: {}
        NumBadEpoch: {} <= {}
        Coef: {}
        Times Reduce = {}
        """.format(current, self.best, self.num_bad_epochs, 0, 0, 0)

    def step(self, metrics, epoch, batch_iteration, step_size):
        if metrics is None or metrics <= 0:
            if batch_iteration is None: batch_iteration = self.last_batch_iteration + 1
            self.last_batch_iteration = batch_iteration
            self.update_lr(step_size)
            return

        if epoch is None:
            epoch = self.last_epoch + 1
        assert epoch == self.last_epoch
        # if batch_iteration is None:
        #     batch_iteration = self.last_batch_iteration + 1
        if batch_iteration is not None: self.last_batch_iteration = batch_iteration
        self.update_lr(step_size)
        return

    def _set_lr(self, lrs):
        # for i, param_group in enumerate(self.optimizer.param_groups):
        #     param_group['lr'] = lr
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr

    def update_lr(self, step_size=None):
        pass

    def _init_is_better(self, eval_mode, threshold, threshold_mode):
        if eval_mode not in {'min', 'max'}:
            raise ValueError('mode ' + eval_mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if eval_mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.is_better = partial(self._cmp, eval_mode, threshold_mode, threshold)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key not in {'optimizer', 'is_better', 'scale_fn'}}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(eval_mode=self.eval_mode, threshold=self.threshold, threshold_mode=self.threshold_mode)

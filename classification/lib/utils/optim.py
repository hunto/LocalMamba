import torch
import torch.optim as optim


def build_optimizer(opt, model, lr, eps=1e-10, momentum=0.9, weight_decay=1e-5, filter_bias_and_bn=True, nesterov=True, sort_params=False):
    # params in dyrep must be sorted to make sure optimizer can correctly
    # load the states in resuming
    params = get_params(model, lr, weight_decay, filter_bias_and_bn, sort_params=sort_params)

    if opt == 'rmsprop':
        optimizer = optim.RMSprop(params, lr, eps=eps, weight_decay=weight_decay, momentum=momentum)
    elif opt == 'rmsproptf':
        optimizer = RMSpropTF(params, lr, eps=eps, weight_decay=weight_decay, momentum=momentum)
    elif opt == 'sgd':
        optimizer = optim.SGD(params, lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    elif opt == 'adamw':
        optimizer = optim.AdamW(params, lr, eps=eps, weight_decay=weight_decay)
    else:
        raise NotImplementedError(f'Optimizer {opt} not implemented.')
    return optimizer


def get_params(model, lr, weight_decay=1e-5, filter_bias_and_bn=True, sort_params=False):
    if weight_decay != 0 and filter_bias_and_bn:
        if hasattr(model, 'no_weight_decay'):
            skip_list = model.no_weight_decay()
            print(f'no weight decay: {skip_list}')
        else:
            skip_list = ()
        params = _add_weight_decay(model, lr, weight_decay, skip_list=skip_list, sort_params=sort_params)
        weight_decay = 0
    else:
        named_params = list(model.named_parameters())
        if sort_params:
            named_params.sort(key=lambda x: x[0])
        params = [x[1] for x in named_params]
        params = [{'params': params, 'initial_lr': lr}]
    return params


def _add_weight_decay(model, lr, weight_decay=1e-5, skip_list=(), sort_params=False):
    decay = []
    no_decay = []
    named_params = list(model.named_parameters())
    if sort_params:
        named_params.sort(key=lambda x: x[0])
    for name, param in named_params:
        if not param.requires_grad:
            continue  # frozen weights
        skip = False
        for skip_name in skip_list:
            if skip_name.startswith('[g]'):
                if skip_name[3:] in name:
                    skip = True
            elif name == skip_name:
                skip = True
        if len(param.shape) == 1 or name.endswith(".bias") or skip:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0., 'initial_lr': lr},
        {'params': decay, 'weight_decay': weight_decay, 'initial_lr': lr}]


class RMSpropTF(optim.Optimizer):
    """Implements RMSprop algorithm (TensorFlow style epsilon)
    NOTE: This is a direct cut-and-paste of PyTorch RMSprop with eps applied before sqrt
    and a few other modifications to closer match Tensorflow for matching hyper-params.
    Noteworthy changes include:
    1. Epsilon applied inside square-root
    2. square_avg initialized to ones
    3. LR scaling of update accumulated in momentum buffer
    Proposed by G. Hinton in his
    `course <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.
    The centered version first appears in `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        alpha (float, optional): smoothing (decay) constant (default: 0.9)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        decoupled_decay (bool, optional): decoupled weight decay as per https://arxiv.org/abs/1711.05101
        lr_in_momentum (bool, optional): learning rate scaling is included in the momentum buffer
            update as per defaults in Tensorflow
    """

    def __init__(self, params, lr=1e-2, alpha=0.9, eps=1e-10, weight_decay=0, momentum=0., centered=False,
                 decoupled_decay=False, lr_in_momentum=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay,
                        decoupled_decay=decoupled_decay, lr_in_momentum=lr_in_momentum)
        super(RMSpropTF, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RMSpropTF, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

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
                    raise RuntimeError('RMSprop does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.ones_like(p.data)  # PyTorch inits to zero
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                one_minus_alpha = 1. - group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    if 'decoupled_decay' in group and group['decoupled_decay']:
                        p.data.add_(-group['weight_decay'], p.data)
                    else:
                        grad = grad.add(group['weight_decay'], p.data)

                # Tensorflow order of ops for updating squared avg
                square_avg.add_(one_minus_alpha, grad.pow(2) - square_avg)
                # square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)  # PyTorch original

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.add_(one_minus_alpha, grad - grad_avg)
                    # grad_avg.mul_(alpha).add_(1 - alpha, grad)  # PyTorch original
                    avg = square_avg.addcmul(-1, grad_avg, grad_avg).add(group['eps']).sqrt_()  # eps moved in sqrt
                else:
                    avg = square_avg.add(group['eps']).sqrt_()  # eps moved in sqrt

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    # Tensorflow accumulates the LR scaling in the momentum buffer
                    if 'lr_in_momentum' in group and group['lr_in_momentum']:
                        buf.mul_(group['momentum']).addcdiv_(group['lr'], grad, avg)
                        p.data.add_(-buf)
                    else:
                        # PyTorch scales the param update by LR
                        buf.mul_(group['momentum']).addcdiv_(grad, avg)
                        p.data.add_(-group['lr'], buf)
                else:
                    p.data.addcdiv_(-group['lr'], grad, avg)

        return loss



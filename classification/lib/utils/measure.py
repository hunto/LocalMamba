import torch


def get_params(model, ignore_auxiliary_head=True):
    if not ignore_auxiliary_head:
        params = sum([m.numel() for m in model.parameters()])
    else:
        params = sum([m.numel() for k, m in model.named_parameters() if 'auxiliary_head' not in k])
    return params

def get_flops(model, input_shape=(3, 224, 224)):
    if hasattr(model, 'flops'):
        return model.flops(input_shape)
    else:
        return get_flops_hook(model, input_shape)

def get_flops_hook(model, input_shape=(3, 224, 224)):
    is_training = model.training
    list_conv = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        assert self.in_channels % self.groups == 0

        kernel_ops = self.kernel_size[0] * self.kernel_size[
            1] * (self.in_channels // self.groups)
        params = output_channels * kernel_ops
        flops = batch_size * params * output_height * output_width

        list_conv.append(flops)

    list_linear = []

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement()

        flops = batch_size * weight_ops
        list_linear.append(flops)

    def foo(net, hook_handle):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                hook_handle.append(net.register_forward_hook(conv_hook))
            if isinstance(net, torch.nn.Linear):
                hook_handle.append(net.register_forward_hook(linear_hook))
            return
        for c in childrens:
            foo(c, hook_handle)

    hook_handle = []
    foo(model, hook_handle)
    input = torch.rand(*input_shape).unsqueeze(0).to(next(model.parameters()).device)
    model.eval()
    with torch.no_grad():
        out = model(input)
    for handle in hook_handle:
        handle.remove()

    total_flops = sum(sum(i) for i in [list_conv, list_linear])
    model.train(is_training)
    return total_flops


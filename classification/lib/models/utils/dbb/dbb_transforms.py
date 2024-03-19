import numpy as np
import torch
import torch.nn.functional as F


def restore_bn(kernel, bn, conv_bias):
    gamma = bn.weight
    std = (bn.running_var + bn.eps).sqrt()
    bias = -bn.running_mean
    new_bias = (conv_bias - bn.bias) / gamma * std - bias
    new_weight = kernel * (std / gamma).reshape(-1, 1, 1, 1)
    return new_weight, new_bias


def transI_fusebn(kernel, bn, conv_bias):
    gamma = bn.weight
    std = (bn.running_var + bn.eps).sqrt()
    bias = -bn.running_mean
    if conv_bias is not None:
        bias += conv_bias
    return kernel * (
        (gamma / std).reshape(-1, 1, 1, 1)), bn.bias + bias * gamma / std


def transII_addbranch(kernels, biases):
    return torch.sum(kernels, dim=0), torch.sum(biases, dim=0)


def transIII_1x1_kxk(k1, b1, k2, b2, groups=1):
    if groups == 1:
        k = F.conv2d(k2, k1.permute(1, 0, 2, 3))
        b_hat = (k2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3))
    else:
        k_slices = []
        b_slices = []
        k1_T = k1.permute(1, 0, 2, 3)
        k1_group_width = k1.size(0) // groups
        k2_group_width = k2.size(0) // groups
        for g in range(groups):
            k1_T_slice = k1_T[:, g * k1_group_width:(g + 1) *
                              k1_group_width, :, :]
            k2_slice = k2[g * k2_group_width:(g + 1) * k2_group_width, :, :, :]
            k_slices.append(F.conv2d(k2_slice, k1_T_slice))
            b_slices.append(
                (k2_slice *
                 b1[g * k1_group_width:(g + 1) * k1_group_width].reshape(
                     1, -1, 1, 1)).sum((1, 2, 3)))
        k, b_hat = transIV_depthconcat(k_slices, b_slices)
    return k, b_hat + b2


def transIV_depthconcat(kernels, biases):
    return torch.cat(kernels), torch.cat(biases)


def transV_avg(channels, kernel_size, groups):
    input_dim = channels // groups
    k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
    k[np.arange(channels).tolist(),
      np.tile(np.arange(input_dim), groups).tolist(
      ), :, :] = 1.0 / kernel_size**2
    return k


def transVI_multiscale(kernel, target_kernel_size):
    """
    NOTE: This has not been tested with non-square kernels
        (kernel.size(2) != kernel.size(3)) nor even-size kernels
    """
    W_pixels_to_pad = (target_kernel_size - kernel.size(2)) // 2
    H_pixels_to_pad = (target_kernel_size - kernel.size(3)) // 2
    return F.pad(
        kernel,
        [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad])


def transVII_kxk_1x1(k1, b1, k2, b2):
    return F.conv2d(k1.permute(1, 0, 2, 3),
                    k2).permute(1, 0, 2,
                                3), (k2 * b1.reshape(-1, 1, 1, 1)).sum(
                                    (1, 2, 3)) + b2


def transIIX_kxk_kxk(k1, b1, k2, b2, groups=1):
    k1 = torch.from_numpy(
        np.flip(np.flip(np.array(k1), axis=3), axis=2).copy())
    k_size = k1.size(2)
    padding = k_size // 2 + 1
    if groups == 1:
        k = F.conv2d(k2, k1.permute(1, 0, 2, 3), padding=padding)
        b_hat = (k2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3))
    else:
        k_slices = []
        b_slices = []
        k1_T = k1.permute(1, 0, 2, 3)
        k1_group_width = k1.size(0) // groups
        k2_group_width = k2.size(0) // groups
        for g in range(groups):
            k1_T_slice = k1_T[:, g * k1_group_width:(g + 1) *
                              k1_group_width, :, :]
            k2_slice = k2[g * k2_group_width:(g + 1) * k2_group_width, :, :, :]
            k_slices.append(F.conv2d(k2_slice, k1_T_slice, padding=padding))
            b_slices.append(
                (k2_slice *
                 b1[g * k1_group_width:(g + 1) * k1_group_width].reshape(
                     1, -1, 1, 1)).sum((1, 2, 3)))
        k, b_hat = transIV_depthconcat(k_slices, b_slices)
    return k, b_hat + b2


def transIX_bn_to_1x1(bn, in_channels, groups=1):
    input_dim = in_channels // groups
    kernel_value = np.zeros((in_channels, input_dim, 3, 3), dtype=np.float32)
    for i in range(in_channels):
        kernel_value[i, i % input_dim, 1, 1] = 1
    id_tensor = torch.from_numpy(kernel_value).to(bn.weight.device)
    kernel = id_tensor
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


import torch.nn as nn
import torch.nn.functional as F


class KLDivergence(nn.Module):
    """A measure of how one probability distribution Q is different from a
    second, reference probability distribution P.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        reduction (str): Specifies the reduction to apply to the loss:
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied,
            ``'batchmean'``: the sum of the output will be divided by
                the batchsize,
            ``'sum'``: the output will be summed,
            ``'mean'``: the output will be divided by the number of
                elements in the output.
            Default: ``'batchmean'``
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(
        self,
        tau=1.0,
        reduction='batchmean',
    ):
        super(KLDivergence, self).__init__()
        self.tau = tau

        accept_reduction = {'none', 'batchmean', 'sum', 'mean'}
        assert reduction in accept_reduction, \
            f'KLDivergence supports reduction {accept_reduction}, ' \
            f'but gets {reduction}.'
        self.reduction = reduction

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W) or shape (N, C).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W) or shape (N, C).

        Return:
            torch.Tensor: The calculated loss value.
        """
        preds_T = preds_T.detach()
        softmax_pred_T = F.softmax(preds_T / self.tau, dim=1)
        logsoftmax_preds_S = F.log_softmax(preds_S / self.tau, dim=1)
        loss = (self.tau**2) * F.kl_div(
            logsoftmax_preds_S, softmax_pred_T, reduction=self.reduction)
        return loss

import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    # Formula Given above.
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        inputs = F.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).mean()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.mean() + targets.mean() + smooth)
        Dice_BCE = 0.9*BCE + 0.1*dice_loss

        return Dice_BCE.mean()
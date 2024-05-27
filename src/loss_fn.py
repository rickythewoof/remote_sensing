import torch

import torch.nn.functional as F

class CombinedLoss(torch.nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets)
        dice_loss = self.dice_loss(inputs, targets)
        combined_loss = 0.25 * bce_loss + 0.75 * dice_loss
        return combined_loss

    def dice_loss(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs)
        intersection = torch.sum(inputs * targets)
        union = torch.sum(inputs) + torch.sum(targets)
        dice_score = (2 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice_score
        return dice_loss
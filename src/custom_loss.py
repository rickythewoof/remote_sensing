import torch

import torch.nn.functional as F

class BCEDiceLoss(torch.nn.Module):
    def __init__(self, proportion=0.5):
        super(BCEDiceLoss, self).__init__()
        self.proportion = proportion

    def forward(self, y_pred, y_true):

        y_pred = torch.sigmoid(y_pred)
        bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true)

        # Dice Loss
        smooth = 1e-5
        intersection = torch.sum(y_pred * y_true)
        union = torch.sum(y_pred) + torch.sum(y_true)
        dice_loss = 1 - (2 * intersection + smooth) / (union + smooth)

        # Combine BCE and Dice Loss
        loss = self.proportion*bce_loss + (1-self.proportion)*dice_loss
        
        return loss
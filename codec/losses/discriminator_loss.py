import torch
import torch.nn as nn

from losses.basic_loss import MSEDLoss


class BasicDiscriminatorLoss(nn.Module):
    """Least-square GAN loss."""

    def __init__(self, config=None):
        super(BasicDiscriminatorLoss, self).__init__()

    def forward(self, real_outputs, fake_outputs):
        loss = 0
        real_losses = []
        fake_losses = []
        for dr, dg in zip(real_outputs, fake_outputs):
            dr = dr.float()
            dg = dg.float()
            real_loss = torch.mean((1-dr)**2)
            fake_loss = torch.mean(dg**2)
            loss += (real_loss + fake_loss)
            real_losses.append(real_loss.item())
            fake_losses.append(fake_loss.item())

        return loss


class MSEDiscriminatorLoss(BasicDiscriminatorLoss):
    def __init__(self, config=None):
        super().__init__(config)
        self.mse_loss = MSEDLoss()

    def apply_d_loss(self, scores_fake, scores_real, loss_func):
        total_loss = 0
        total_real_loss = 0
        total_fake_loss = 0
        if isinstance(scores_fake, list):
            # multi-scale loss
            for score_fake, score_real in zip(scores_fake, scores_real):
                loss, real_loss, fake_loss = loss_func(score_fake=score_fake, score_real=score_real)
                total_loss = total_loss + loss
                total_real_loss = total_real_loss + real_loss
                total_fake_loss = total_fake_loss + fake_loss
            # normalize loss values with number of scales
            total_loss /= len(scores_fake)
            total_real_loss /= len(scores_real)
            total_fake_loss /= len(scores_fake)
        else:
            # single scale loss
            total_loss, total_real_loss, total_fake_loss = loss_func(scores_fake, scores_real)
        return total_loss, total_real_loss, total_fake_loss

    def forward(self, real_scores, fake_scores):
        mse_D_loss, mse_D_real_loss, mse_D_fake_loss = self.apply_d_loss(
            scores_fake=fake_scores,
            scores_real=real_scores,
            loss_func=self.mse_loss)
        return mse_D_loss

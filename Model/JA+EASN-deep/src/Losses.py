import torch
import torch.nn as nn
import math
import pytorch_msssim


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, distortion='mse', lmbda=1e-2):
        super().__init__()
        self.distortion = distortion
        self.mse = nn.MSELoss()
        self.msssim = pytorch_msssim.MS_SSIM(data_range=1.0)
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        if self.distortion == 'mse':
            out["dist_loss"] = self.mse(output["x_hat"], target)
            out["loss"] = self.lmbda * 255 ** 2 * out["dist_loss"] + out["bpp_loss"]
        elif self.distortion == 'ms-ssim':
            out["dist_loss"] = self.msssim(output["x_hat"], target)
            out["loss"] = self.lmbda * (1 - out["dist_loss"]) + out["bpp_loss"]
        else:
            raise ValueError(f"Distortion should be 'mse' or 'ms-ssim'. {self.distortion} is not available.")

        return out


import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftHistogram(nn.Module):
    def __init__(self, bins=64, vmin=0.0, vmax=1.0, sigma=0.02, eps=1e-8):
        super().__init__()
        self.bins = bins
        self.vmin = vmin
        self.vmax = vmax
        self.sigma = sigma
        self.eps = eps
        centers = torch.linspace(vmin, vmax, bins)
        self.register_buffer('centers', centers)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.clamp(self.vmin, self.vmax)

        x_flat = x.view(B, C, -1, 1)
        c = self.centers.view(1, 1, 1, self.bins).to(x.device)
        k = torch.exp(-0.5 * ((x_flat - c) / (self.sigma + 1e-12))**2)
        k = k / (k.sum(dim=-1, keepdim=True) + 1e-12)

        hist = k.mean(dim=2)
        hist = hist / (hist.sum(dim=-1, keepdim=True) + 1e-12)
        return hist
        

class ColorHistogramKLLoss(nn.Module):
    def __init__(self, num_bins=64):
        super(ColorHistogramKLLoss, self).__init__()
        self.num_bins = num_bins
        self.compute_histogram = SoftHistogram(self.num_bins)

    def forward(self, img1, img2):
        # Assumes img1 and img2 are normalized to [0, 1]
        hist1 = self.compute_histogram(img1)
        hist2 = self.compute_histogram(img2)
        
        # Add small epsilon to prevent log(0)
        hist1 = hist1 + 1e-8
        hist2 = hist2 + 1e-8

        # Compute the log of hist1 (since KL divergence requires log probabilities)
        hist1_log = hist1.log()

        # Calculate KL divergence
        loss = F.kl_div(hist1_log, hist2, reduction='batchmean') / img1.shape[0]
        return loss

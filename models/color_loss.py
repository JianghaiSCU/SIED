import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorHistogramKLLoss(nn.Module):
    def __init__(self, num_bins=256):
        super(ColorHistogramKLLoss, self).__init__()
        self.num_bins = num_bins

    def compute_histogram(self, img, num_bins):
        # img: (batch_size, channels, height, width), assumes 3 channels for RGB
        batch_size, channels, height, width = img.size()
        histograms = []
        for i in range(channels):
            channel_data = img[:, i, :, :].reshape(batch_size, -1)
            # Compute histogram for each channel
            hist = torch.histc(channel_data, bins=num_bins, min=0.0, max=1.0)
            hist = hist / (hist.sum() + 1e-8)  # Normalize histogram, avoid divide by zero
            histograms.append(hist)
        
        # Stack histograms together
        histograms = torch.stack(histograms, dim=1)  # (batch_size, channels, num_bins)
        return histograms

    def forward(self, img1, img2):
        # Assumes img1 and img2 are normalized to [0, 1]
        hist1 = self.compute_histogram(img1, self.num_bins)
        hist2 = self.compute_histogram(img2, self.num_bins)
        
        # Add small epsilon to prevent log(0)
        hist1 = hist1 + 1e-8
        hist2 = hist2 + 1e-8

        # Compute the log of hist1 (since KL divergence requires log probabilities)
        hist1_log = hist1.log()

        # Calculate KL divergence
        loss = F.kl_div(hist1_log, hist2, reduction='batchmean')
        return loss
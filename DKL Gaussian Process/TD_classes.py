import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeDistributedConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(TimeDistributedConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        # x shape: [batch, time, channels, height, width]
        batch_size, time_steps, C, H, W = x.size()
        # Combine batch and time dimensions
        x = x.view(batch_size * time_steps, C, H, W)
        # Apply convolution
        x = self.conv(x)
        # Separate batch and time dimensions
        x = x.view(batch_size, time_steps, -1, H, W)  # -1 infers the correct number of channels
        return x
    
    
class TimeDistributedAvgPool2D(nn.Module):
    def __init__(self, pool_size):
        super(TimeDistributedAvgPool2D, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        # x shape: [batch, time, channels, height, width]
        batch_size, time_steps, C, H, W = x.size()
        x = x.view(batch_size * time_steps, C, H, W)
        x = F.avg_pool2d(x, self.pool_size)
        _, C, H, W = x.size()
        x = x.view(batch_size, time_steps, C, H, W)
        return x

class TimeDistributedGlobalAvgPool2D(nn.Module):
    def forward(self, x):
        # x shape: [batch, time, channels, height, width]
        batch_size, time_steps, C, H, W = x.size()
        x = x.view(batch_size * time_steps, C, H, W)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(batch_size, time_steps, C, -1)
        return x.squeeze(-1).squeeze(-1)
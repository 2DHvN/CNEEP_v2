import torch

def add_xy_channels(x):
    """
    x: (B, C, H, W)
    return: (B, C+2, H, W)
    """
    B, C, H, W = x.shape
    device = x.device

    # y: (H, W), x: (H, W)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij"
    )

    # (1, 1, H, W) -> (B, 1, H, W)
    xx = xx.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)
    yy = yy.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)

    return torch.cat([x, xx, yy], dim=1)
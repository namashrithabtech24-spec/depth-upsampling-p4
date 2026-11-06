import torch
import torch.nn.functional as F
import cv2
import numpy as np

def tto_upsample(rgb, lr_depth, init_depth, scale, steps=30, lr=0.01):
    H, W = rgb.shape[:2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    rgb_t = torch.tensor(rgb, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0) / 255.0
    lr_t = torch.tensor(lr_depth, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    up_init = torch.tensor(init_depth, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    
    residual = torch.zeros((1, 1, H, W), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([residual], lr=lr)
    
    # BLUR THE RGB to prevent texture copying
    rgb_gray = torch.mean(rgb_t, dim=1, keepdim=True)
    blur_kernel = torch.ones(1, 1, 5, 5, device=device) / 25.0
    rgb_gray_blurred = F.conv2d(rgb_gray, blur_kernel, padding=2)
    
    for i in range(steps):
        optimizer.zero_grad()
        smooth_depth = up_init + residual
        
        down_pred = F.interpolate(smooth_depth, size=(lr_depth.shape[0], lr_depth.shape[1]), mode='area')
        loss_consist = F.l1_loss(down_pred, lr_t)
        
        dx_depth = torch.abs(smooth_depth[:, :, :, 1:] - smooth_depth[:, :, :, :-1])
        dy_depth = torch.abs(smooth_depth[:, :, 1:, :] - smooth_depth[:, :, :-1, :])
        # Use the blurred RGB for edge loss
        dx_rgb = torch.abs(rgb_gray_blurred[:, :, :, 1:] - rgb_gray_blurred[:, :, :, :-1])
        dy_rgb = torch.abs(rgb_gray_blurred[:, :, 1:, :] - rgb_gray_blurred[:, :, :-1, :])
        
        loss_edge = F.l1_loss(dx_depth, dx_rgb) + F.l1_loss(dy_depth, dy_rgb)
        
        # Stronger regularization to keep the residual small
        loss_reg = torch.mean(torch.abs(residual)) * 20 
        
        loss = loss_consist + 0.2 * loss_edge + loss_reg
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        final_out = up_init + residual
        
    return np.clip(final_out.squeeze().cpu().numpy(), 0, 1), residual.detach().cpu().numpy() 


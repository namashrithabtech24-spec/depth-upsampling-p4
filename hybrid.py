import cv2
import numpy as np
from tto import tto_upsample

def guided_filter_upsample(rgb, lr, size_hw, radius=8, eps=1e-3):
    up = cv2.resize(lr, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_CUBIC)
    guide = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    up = up.astype(np.float32)
    out = cv2.ximgproc.guidedFilter(guide, up, radius, eps)
    return np.clip(out, 0, 1)

# FIXED: Added "steps=30" as a parameter so the UI slider can control it
def adaptive_fusion(rgb, lr_depth, scale, steps=30):
    H, W = rgb.shape[:2]
    
    # 1. Classical fills the holes
    gf_out = guided_filter_upsample(rgb, lr_depth, (H, W))
    
    # 2. AI sharpens the edges (residual learning)
    # FIXED: Passed the "steps" variable down to the TTO network
    tto_out, residual = tto_upsample(rgb, lr_depth, gf_out, scale, steps=steps)
    tto_out = np.clip(tto_out, 0, 1)
    
    # 3. Create Trust Map from RGB edges
    rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(rgb_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(rgb_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    
    mag_norm = mag / (mag.max() + 1e-8)
    # Trust the AI sharpening only where RGB has strong edges
    trust_map = 1.0 / (1.0 + np.exp(-15 * (mag_norm - 0.15)))
    
    # 4. Blend
    fused_out = (trust_map * tto_out) + ((1 - trust_map) * gf_out)
    
    return fused_out, trust_map, tto_out, gf_out
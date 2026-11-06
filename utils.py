import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

def compute_metrics(pred, gt):
    diff = pred.astype(np.float64) - gt.astype(np.float64)
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    psnr = float(20 * np.log10(1.0 / (rmse + 1e-12)))
    ssim_val = float(ssim(gt, pred, data_range=1.0))
    return {"RMSE": rmse, "MAE": mae, "PSNR": psnr, "SSIM": ssim_val}

def boundary_f1(pred, gt, thresh=0.1, tol=3):
    def edges(x):
        x = x.astype(np.float32) # <-- FIX ADDED HERE
        gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        return (mag > thresh * (mag.max() + 1e-12)).astype(np.uint8)
    ge, pe = edges(gt), edges(pred)
    k = np.ones((tol*2+1, tol*2+1), np.uint8)
    g_dil = cv2.dilate(ge, k).astype(bool)
    p_dil = cv2.dilate(pe, k).astype(bool)
    tp = np.sum(pe.astype(bool) & g_dil)
    fp = np.sum(pe.astype(bool) & ~g_dil)
    fn = np.sum(ge.astype(bool) & ~p_dil)
    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)
    return float(2 * prec * rec / (prec + rec + 1e-12))

def edge_rmse(pred, gt, thresh=0.1):
    gt_f32 = gt.astype(np.float32) # <-- FIX ADDED HERE
    gx = cv2.Sobel(gt_f32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gt_f32, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    mask = mag > thresh * (mag.max() + 1e-12)
    if mask.sum() < 10:
        return float("nan")
    d = pred[mask] - gt[mask]
    return float(np.sqrt(np.mean(d**2)))
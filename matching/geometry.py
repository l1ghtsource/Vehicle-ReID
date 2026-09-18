import cv2
import numpy as np


def ransac_inliers(keypoints0, keypoints1, reproj=3.0, seed=0):
    pts0 = np.asarray(keypoints0, dtype=np.float32).reshape(-1, 2)
    pts1 = np.asarray(keypoints1, dtype=np.float32).reshape(-1, 2)
    if pts0.shape != pts1.shape:
        raise ValueError("keypoint arrays must have the same shape")
    if float(reproj) <= 0:
        raise ValueError("reproj must be > 0")
    if len(pts0) < 4:
        return np.zeros(len(pts0), dtype=bool)
    cv2.setRNGSeed(int(seed))
    _, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, float(reproj))
    if mask is None:
        return np.zeros(len(pts0), dtype=bool)
    return np.asarray(mask).reshape(-1).astype(bool)


def match_stats(keypoints0, keypoints1, scores, reproj=3.0, seed=0):
    pts0 = np.asarray(keypoints0, dtype=np.float32).reshape(-1, 2)
    pts1 = np.asarray(keypoints1, dtype=np.float32).reshape(-1, 2)
    sc = np.asarray(scores, dtype=np.float32).reshape(-1)
    if not (len(pts0) == len(pts1) == len(sc)):
        raise ValueError("keypoints and scores must align")
    inliers = ransac_inliers(pts0, pts1, reproj=reproj, seed=seed)
    n = len(sc)
    n_in = int(inliers.sum())
    return {
        "n_matches": n,
        "n_inliers": n_in,
        "inlier_ratio": n_in / max(n, 1),
        "score_sum": float(sc.sum()) if n else 0.0,
        "score_mean": float(sc.mean()) if n else 0.0,
        "score_max": float(sc.max()) if n else 0.0,
        "inliers": inliers,
    }

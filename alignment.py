import math
import random
import cv2
import numpy as np

eTranslate = 0
eHomography = 1


def computeHomography(f1, f2, matches, A_out=None):
    num_matches = len(matches)
    num_rows = 2 * num_matches
    num_cols = 9
    A = np.zeros((num_rows, num_cols))

    # ---------- TODO 1 : 组装线性方程组 A ----------
    for i, m in enumerate(matches):
        (a_x, a_y) = f1[m.queryIdx].pt  # point in img 1
        (b_x, b_y) = f2[m.trainIdx].pt  # point in img 2

        row = 2 * i
        A[row,     :] = [-a_x, -a_y, -1, 0, 0, 0, a_x * b_x, a_y * b_x, b_x]
        A[row + 1, :] = [0, 0, 0, -a_x, -a_y, -1, a_x * b_y, a_y * b_y, b_y]
    # ------------------------------------------------

    U, s, Vt = np.linalg.svd(A)
    if A_out is not None:
        A_out[:] = A

    # ---------- TODO 2 : 由 SVD 得到单应矩阵 ----------
    h = Vt[-1, :]                  # 最小奇异值对应的特征向量
    H = h.reshape((3, 3))
    H = H / H[2, 2]                # 归一化
    # -------------------------------------------------
    return H


def getInliers(f1, f2, matches, M, RANSACthresh):
    inlier_indices = []
    for i, m in enumerate(matches):
        (a_x, a_y) = f1[m.queryIdx].pt
        (b_x, b_y) = f2[m.trainIdx].pt

        p1 = np.array([a_x, a_y, 1.0])
        p2_est = M @ p1
        p2_est /= p2_est[2]
        dist = np.linalg.norm(p2_est[:2] - np.array([b_x, b_y]))
        if dist <= RANSACthresh:
            inlier_indices.append(i)
    return inlier_indices


def leastSquaresFit(f1, f2, matches, inlier_indices):
    if len(inlier_indices) < 4:
        return np.eye(3)
    inlier_matches = [matches[i] for i in inlier_indices]
    M = computeHomography(f1, f2, inlier_matches)
    return M


def alignPair(f1, f2, matches, nRANSAC, RANSACthresh):
    # ---------- TODO 3 : RANSAC 主循环 ----------
    best_inliers = []
    best_H = np.eye(3)

    if len(matches) < 4:
        return best_H

    for _ in range(nRANSAC):
        sample_ids = random.sample(range(len(matches)), 4)
        sample_matches = [matches[i] for i in sample_ids]
        H_candidate = computeHomography(f1, f2, sample_matches)

        inliers = getInliers(f1, f2, matches, H_candidate, RANSACthresh)
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H_candidate
    # 以全部内点做一次最小二乘精化
    M = leastSquaresFit(f1, f2, matches, best_inliers)
    # ------------------------------------------------
    return M

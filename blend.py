import math
import cv2
import numpy as np


class ImageInfo:
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position


def imageBoundingBox(img, M):
    # ---------- TODO 6 : 计算单幅图像在全景中的包围盒 ----------
    h, w = img.shape[:2]
    corners = np.array([[0, 0, 1],
                        [w, 0, 1],
                        [w, h, 1],
                        [0, h, 1]]).T          # 3×4
    trans_corners = M @ corners
    trans_corners /= trans_corners[2, :]
    xs, ys = trans_corners[0, :], trans_corners[1, :]
    minX, minY = xs.min(), ys.min()
    maxX, maxY = xs.max(), ys.max()
    # ------------------------------------------------------------
    return int(minX), int(minY), int(maxX), int(maxY)


def accumulateBlend(img, acc, M, blendWidth):
    # ---------- TODO 7 : 加权累积图像 ----------
    h, w = img.shape[:2]

    # 1) 生成水平帽函数权重
    weights = np.ones((h, w), dtype=np.float32)
    if blendWidth > 0:
        ramp = np.linspace(0, 1, blendWidth, dtype=np.float32)
        weights[:, :blendWidth] = ramp
        weights[:, -blendWidth:] = ramp[::-1]

    # 2) 透视变换到累加器坐标系
    size = (acc.shape[1], acc.shape[0])        # (W, H)
    warped_img = cv2.warpPerspective(img.astype(np.float32), M, size)
    warped_w   = cv2.warpPerspective(weights, M, size)

    # 3) 累积：前三通道存颜色，加第 4 通道存权重
    for c in range(3):
        acc[:, :, c] += warped_img[:, :, c] * warped_w
    acc[:, :, 3] += warped_w
    # -----------------------------------------------------------


def normalizeBlend(acc):
    # ---------- TODO 8 : 归一化得到最终图像 ----------
    img = np.zeros(acc.shape[:3], dtype=np.uint8)
    w = acc[:, :, 3]
    nonzero = w > 0
    img[nonzero] = (acc[nonzero, :3] / w[nonzero, None]).astype(np.uint8)
    # -----------------------------------------------------------
    return img


def getAccSize(ipv):
    # 计算全景大包围盒
    minX = np.inf
    minY = np.inf
    maxX = -np.inf
    maxY = -np.inf
    channels = -1
    width = -1            # 假设所有输入宽度相同
    for i in ipv:
        img = i.img
        if channels == -1:
            channels = img.shape[2]
            width = img.shape[1]
        minx_i, miny_i, maxx_i, maxy_i = imageBoundingBox(img, i.position)
        # ---------- TODO 9 : 更新整体包围盒 ----------
        minX = min(minX, minx_i)
        minY = min(minY, miny_i)
        maxX = max(maxX, maxx_i)
        maxY = max(maxY, maxy_i)
        # ---------------------------------------------------

    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    print('accWidth, accHeight:', (accWidth, accHeight))
    translation = np.array([[1, 0, -minX],
                            [0, 1, -minY],
                            [0, 0, 1]])
    return accWidth, accHeight, channels, width, translation


def pasteImages(ipv, translation, blendWidth, accWidth, accHeight, channels):
    acc = np.zeros((accHeight, accWidth, channels + 1), dtype=np.float32)
    for i in ipv:
        M_trans = translation @ i.position
        accumulateBlend(i.img, acc, M_trans, blendWidth)
    return acc


def getDriftParams(ipv, translation, width):
    for count, i in enumerate(ipv):
        if count not in (0, len(ipv) - 1):
            continue
        M_trans = translation @ i.position
        p = M_trans @ np.array([0.5 * width, 0, 1])
        if count == 0:
            x_init, y_init = p[:2] / p[2]
        else:
            x_final, y_final = p[:2] / p[2]
    return x_init, y_init, x_final, y_final


def computeDrift(x_init, y_init, x_final, y_final, width):
    A = np.identity(3)
    drift = float(y_final - y_init)
    length = float(x_final - x_init)
    A[0, 2] = -0.5 * width                # 左移半幅
    A[1, 0] = -drift / length             # 去除垂直漂移
    return A


def blendImages(ipv, blendWidth, is360=False, A_out=None):
    accW, accH, channels, width, translation = getAccSize(ipv)
    acc = pasteImages(ipv, translation, blendWidth, accW, accH, channels)
    compImage = normalizeBlend(acc)

    outputWidth = (accW - width) if is360 else accW
    x_i, y_i, x_f, y_f = getDriftParams(ipv, translation, width)

    # ---------- TODO 10 : 生成裁剪 / 漂移修正矩阵 ----------
    if is360 and len(ipv) > 1:
        A = computeDrift(x_i, y_i, x_f, y_f, width)
    else:
        A = np.identity(3)  # 仅裁剪，不修正漂移
    # --------------------------------------------------------

    if A_out is not None:
        A_out[:] = A

    cropped = cv2.warpPerspective(compImage, A, (outputWidth, accH),
                                  flags=cv2.INTER_LINEAR)
    return cropped

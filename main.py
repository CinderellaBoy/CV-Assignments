import argparse
import os
import cv2
import numpy as np

import alignment
import blend

DEFAULT_FOCAL_LENGTH = 678
DEFAULT_K1 = -0.21
DEFAULT_K2 = 0.26


# ---------- TODO 11 : 解析命令行默认值 ----------
def parse_args():
    parser = argparse.ArgumentParser(description="Panorama Maker (Command Line)")
    parser.add_argument("--output", "-o", type=str,
                        default="results/panorama.jpg",
                        help="Path to save the output panorama image.")
    parser.add_argument("--blend_width", type=int, default=64,
                        help="Width of the blending region in pixels.")
    parser.add_argument("--ransac_rounds", type=int, default=1000,
                        help="Number of RANSAC iterations for alignment.")
    parser.add_argument("--ransac_thresh", type=float, default=3.0,
                        help="RANSAC distance threshold for inliers.")
    parser.add_argument("--focal_length", type=float, default=DEFAULT_FOCAL_LENGTH,
                        help="Focal length of the camera (in pixels).")
    parser.add_argument("--k1", type=float, default=DEFAULT_K1,
                        help="First radial distortion parameter.")
    parser.add_argument("--k2", type=float, default=DEFAULT_K2,
                        help="Second radial distortion parameter.")
    parser.add_argument("--is_360", action='store_true',
                        help="Indicate if the panorama is 360 degrees.")
    return parser.parse_args()
# -----------------------------------------------------------


def compute_mapping(left_image, right_image, args):
    left_grey = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_grey = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    left_kp, left_desc = orb.detectAndCompute(left_grey, None)
    right_kp, right_desc = orb.detectAndCompute(right_grey, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(left_desc, right_desc)
    matches = sorted(matches, key=lambda x: x.distance)

    n_matches = max(4, int(0.20 * len(matches)))
    matches = matches[:n_matches]

    return alignment.alignPair(left_kp, right_kp, matches,
                               args.ransac_rounds, args.ransac_thresh)


def main():
    args = parse_args()

    # ---------- TODO 12 : 收集输入图像路径 ----------
    src_dir = "source_images"
    input_image_paths = sorted(
        [os.path.join(src_dir, f)
         for f in os.listdir(src_dir)
         if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    )
    # ---------------------------------------------------

    images = [cv2.imread(p) for p in input_image_paths if os.path.exists(p)]
    if len(images) < 2:
        print("Need at least two valid images in 'source_images' folder.")
        return

    t = np.eye(3)
    ipv = []
    for i in range(len(images) - 1):
        print(f"Aligning img {i} -> img {i + 1}")
        ipv.append(blend.ImageInfo(f'img{i}', images[i], np.linalg.inv(t)))
        mapping = compute_mapping(images[i], images[i + 1], args)
        if mapping is None:
            print("Alignment failed – aborting.")
            return
        t = mapping @ t
    ipv.append(blend.ImageInfo(f'img{len(images)-1}', images[-1], np.linalg.inv(t)))

    if args.is_360 and len(images) > 1:
        print("Closing loop for 360 ° panorama …")
        t_loop = compute_mapping(images[-1], images[0], args)
        if t_loop is not None:
            t = t_loop @ t
            ipv.append(blend.ImageInfo('loop', images[0], np.linalg.inv(t)))

    print("Blending …")
    panorama = blend.blendImages(ipv, args.blend_width, args.is_360)

    # ---------- TODO 13 : 去黑边并保存结果 ----------
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(thresh)
    cropped = panorama[y:y + h, x:x + w]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, cropped)
    print(f"Panorama saved to {args.output}")
    # ---------------------------------------------------


if __name__ == "__main__":
    main()

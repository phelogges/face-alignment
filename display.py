import cv2
import numpy as np
import argparse
import os


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, default="lmks_68.txt")
    parser.add_argument("--img_dir", type=str, default="")
    parser.add_argument("--name", type=str, default="")
    return parser.parse_args()


def display(img_name, file_name):
    f = np.loadtxt(file_name, str)
    dict = {}
    for i in f:
        dict[i[0]] = np.clip(i[1:].astype(np.float32), 0., None).reshape(
            (68, 2))
    img = cv2.imread(img_name)
    pts = dict[os.path.basename(img_name)]
    i = 0
    while i < 68:
        cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), (0, 255, 0), 5)
        i += 1
    cv2.imshow("Draw landmarks", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = arg_parser()
    img_name = os.path.join(args.img_dir, args.name)
    file_name = args.file_name
    display(img_name, file_name)

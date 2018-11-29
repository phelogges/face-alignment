import cv2
import numpy as np
import argparse
import os


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r", type=int, default=5, help="radios of circle")
    # parser.add_argument("--img_name", type=str, default="")
    parser.add_argument("--dir_name", type=str, default="")
    parser.add_argument("--lmk_file", type=str, default="")
    return parser.parse_args()


def img_creater(img, lmks, r):
    img = cv2.circle(img, lmks, r, (255, 255), -1)
    return img


if __name__ == "__main__":
    args = arg_parser()
    if not os.path.exists(args.dir_name):
        os.mkdir(args.dir_name)
    f = np.loadtxt(args.lmk_file, str)
    img_name = []
    dict = {}
    for i in f:
        img_name.append(i[0])
        dict[i[0]] = i[1:].astype(np.float32).reshape(
            (68, 2))

    for name in img_name:
        shape = (256, 256)
        img = np.zeros(shape, np.float32) - np.ones(shape, np.float32)
        for pt in dict[name]:
            img = img_creater(img, tuple(pt), args.r)
        img = img / 127.5 - 1.
        cv2.imwrite(os.path.join(args.dir_name, name), img)
        if img_name.index(name) % 500 == 0:
            print(
                "Processed {}/{}".format(img_name.index(name), len(img_name)))
    print("Done")

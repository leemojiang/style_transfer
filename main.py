import argparse

from style_transfer import *
from PIL import Image
import matplotlib.pyplot as plt


def main(args):
    content_img_path = args.content_img
    style_img_path = args.style_img
    output_path = args.output_path

    content_img = Image.open(content_img_path).convert("RGB")
    style_img = Image.open(style_img_path).convert("RGB")

    # plt.imshow(content_img)
    # plt.show()
    # plt.imshow(style_img)
    # plt.show()

    output_img = style_transfer(content_img, style_img, input_img=content_img)
    output_img = output_img.cpu().clone()
    output_img = output_img.squeeze(0)

    output_img = transforms.ToPILImage()(output_img)
    plt.imshow(output_img)
    plt.show()

    if output_path:
        output_img.save(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--content_img")
    parser.add_argument("-s", "--style_img")
    parser.add_argument("-o", "--output_path")
    args = parser.parse_args()
    main(args)

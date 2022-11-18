"""preprocess"""
import os
import argparse

from src.dataset import createDataset


parser = argparse.ArgumentParser(description='preprocesss')

parser.add_argument('--data_dir', type=str, default='')
parser.add_argument("--img_dir", type=str, help="")
parser.add_argument('--category', type=str, default='')

args = parser.parse_args()


if __name__ == '__main__':
    _, ds_test = createDataset(args.data_dir, args.category)

    for i, data in enumerate(ds_test.create_dict_iterator()):
        img = data['img'].asnumpy()

        # save img
        file_name_img = "data_img" + "_" + str(i) + ".bin"
        file_path = os.path.join(args.img_dir, file_name_img)
        img.tofile(file_path)

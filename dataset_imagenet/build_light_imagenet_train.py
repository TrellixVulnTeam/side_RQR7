import os
import argparse
import random
from shutil import copyfile


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='dataset dir')
    parser.add_argument('--num', type=int, help='data number')
    parser.add_argument('--output', type=str, help='new data folder')
    args = parser.parse_args()

    dataset_path = args.dataset
    output_path = args.output
    data_num = args.num
    # Create output folder
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    for d in os.scandir(dataset_path):
        if d.is_dir():
            print('Folder: {}'.format(d.name))
            new_dir = os.path.join(output_path, d.name)
            if not os.path.isdir(new_dir):
                os.mkdir(new_dir)
                files_list = [f.name for f in os.scandir(d.path) if os.path.isfile(f.path)]
                if len(files_list) > data_num:
                    random.shuffle(files_list)
                    for i in range(data_num):
                        filename = files_list[i]
                        file_path = os.path.join(d.path, filename)
                        dst = os.path.join(new_dir, filename)
                        copyfile(file_path, dst)
                else:
                    for filename in files_list:
                        file_path = os.path.join(d.path, filename)
                        dst = os.path.join(new_dir, filename)
                        copyfile(file_path, dst)


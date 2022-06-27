import glob
import piexif
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Remove EXIF file.')

    parser.add_argument('--dir', default=None, type=str, 
                        help='The top directory, like ImageNet that include train and val')

    return parser.parse_args()

def main(args):
    nfiles = 0
    pathname = os.path.join(args.dir, '**/*.JPEG')
    for filename in glob.iglob(pathname, recursive=True):
        nfiles = nfiles + 1
        print("About to process file %d, which is %s." % (nfiles,filename))
        piexif.remove(filename)


if __name__ == '__main__':
    args = parse_args()
    main(args)
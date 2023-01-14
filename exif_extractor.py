# extract meta and compress img for fast processing

from glob import glob 
from os.path import join
from pprint import pprint
import cv2
from tqdm import tqdm
from os.path import basename, expanduser, exists
import os
import json
import exif


def get_exif(path):
    selected_keys = []
    img_exif = None
    exif_info = {}
    with open(img_path, 'rb') as img_file:
        img_exif = exif.Image(img_file)
    if not selected_keys:
        all_exif_keys = img_exif.list_all()
    else:
        all_exif_keys = selected_keys
    for key in all_exif_keys:
        try:
            if type(img_exif.get(key)) in [str, float, int, tuple, None]:
                exif_info[key] = img_exif.get(key)
        except:
            pass
    # pprint(exif_info)

    return exif_info

import argparse

parser = argparse.ArgumentParser(description='Please set input folder and output folder')
parser.add_argument('-i','--input',help='input path')
parser.add_argument('-o','--output',default="~/Desktop/tmp_exif_output",help='result path')

args = parser.parse_args()
folder_path = args.input
cache_path = expanduser(args.output)

print(folder_path)
if not exists(join(cache_path,"thumbnails")):
    os.makedirs(join(cache_path,"thumbnails"))
img_paths = sorted(glob(join(folder_path,"*.jpg")))

# calc better exif & save to local
all_raw_exif = {}
for img_path in tqdm(img_paths):
    # load exif and h,w, use precalc if using betterexif folder instead of original
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    exif_info = get_exif(img_path)
    # add filename
    exif_info['filename'] = basename(img_path)
    exif_info['orig_size'] = (h,w)

    all_raw_exif[basename(img_path)] = exif_info
    # save thumbnails
    h,w = img.shape[:2]
    img_thumb = cv2.resize(img,(int(512/h*w),512))
    cv2.imwrite(join(cache_path,"thumbnails",basename(img_path)),img_thumb)


with open(join(cache_path,"all_raw_exif.json"),"w") as fp:
    json.dump(all_raw_exif, fp, sort_keys=True, indent=4, ensure_ascii=False)
import os, sys
import cv2
from natsort import natsorted
from glob import glob
from PIL import Image, ImageFile
import numpy as np
from tqdm import tqdm
# ImageFile.LOAD_TRUNCATED_IMAGES = True

image_dir = '/local_data/M112cyliang/EEIP_data/train_image_new'
mask_dir = '/local_data/M112cyliang/EEIP_data/train_mask_new'

# files = natsorted(glob(os.path.join(dir, '*.png')))
files = os.listdir(image_dir)

cnt = 0


for f in tqdm(files):
	image_fname = os.path.join(image_dir, f)
	mask_fname = os.path.join(mask_dir, f)
	try:
		img = Image.open(image_fname)
		img.verify()
		img = Image.open(mask_fname)
		img.verify()
	except(IOError, SyntaxError) as e:
		print(e)
		print('Bad file: ', f)
		cnt += 1

		# remove
		os.system("rm -r {}".format(image_fname))
		os.system("rm -r {}".format(mask_fname))


print('total files {}'.format(len(files)))
print('num bad file: {}'.format(cnt))


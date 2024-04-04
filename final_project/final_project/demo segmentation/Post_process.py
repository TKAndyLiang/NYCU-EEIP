import sklearn
import numpy as np
import math
import argparse
import os, sys
from natsort import natsorted
from glob import glob
from PIL import Image, ImageFilter, ImageChops
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from numpy import asarray
from matplotlib import pyplot as plt
from tqdm import tqdm
import skimage
from skimage.morphology import square
from joblib import Parallel, delayed
import multiprocessing


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", type=str, default='test/image', help="directory of the input file (image)")
parser.add_argument("-l", "--label", type=str, default='test/result', help="raw label of the image")
parser.add_argument("-o", "--output", type=str, default="test/post_output", help="output file")
parser.add_argument("-k", "--kernel_size", type=int, nargs=5, default=[5, 3, 3, 3, 3], help="size of structure element in morphological operations")
parser.add_argument("-t", "--threshold", type=float, default=175, help="lightness threshold of sky elimination")
args = parser.parse_args()

POST_DIR = 'test/post_output'

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


## sky segmentation and elimination
def post_process_sky_elimination(input_dir, label_dir, output_dir, threshold=175, kernel_size=[5,7,7,7,7]):
    image_in = Image.open(input_dir).convert('RGB')
    image_in_blurred = image_in.filter(ImageFilter.GaussianBlur(radius=1))
    image_in_blurred_I = image_in_blurred.convert("L")
    image_in_data_I = asarray(image_in_blurred_I)
    image_in_bright = np.where(image_in_data_I > threshold, 255, 0)
    image_in_marker = np.zeros(image_in_bright.shape , dtype=np.uint8)
    index = np.where(image_in_bright == 255)
    for x_axis in range(index[0].size):
        if index[0][x_axis] < image_in_data_I.shape[0]/10:
            image_in_marker[index[0][x_axis]][index[1][x_axis]] = 255
    image_in_marker = Image.fromarray(image_in_marker)
    image_in_mask = Image.fromarray(np.uint8(image_in_bright))
    morphological_size = kernel_size[0]

    im3 = image_in_marker
    im4 = im3.filter(ImageFilter.MaxFilter(morphological_size))
    im4 = ImageChops.multiply(im4, image_in_mask)
    while im4 != im3:
        im3 = im4
        im4 = im3.filter(ImageFilter.MaxFilter(morphological_size))
        im4 = ImageChops.multiply(im4, image_in_mask)
    image_in_sub = im4

    image_label = Image.open(label_dir)
    image_label = image_label.convert("L")
    imk = ImageChops.subtract(image_label,image_in_sub)
    post_process_morphology(image_label, imk, output_dir, kernel_size[1:5])
    # post_process_morphology_skimage(image_label, imk, output_dir, kernel_size[1:5])



def post_process_morphology_skimage(image_label, imk, output_dir, kernel_size=(3,3,3,3)):
    image_label = asarray(image_label).copy()
    imk = asarray(imk).copy()
    image_label = np.where(image_label == 255,1,0)
    imk = np.where(imk == 255,1,0)
    im3 = skimage.morphology.erosion(imk, square(kernel_size[0]))
    for i in range(5):
        im3 = skimage.morphology.erosion(im3, square(kernel_size[0]))
    im4 = skimage.morphology.reconstruction(im3, image_label, method='dilation', footprint=square(kernel_size[1]))
    im2 = im4

    im3 = skimage.morphology.dilation(im2, square(kernel_size[2]))
    for i in range(5):
        im3 = skimage.morphology.dilation(im3, square(kernel_size[2]))
    im4 = skimage.morphology.reconstruction(im3, im2, method='erosion', footprint=square(kernel_size[3]))
    result_label = im4
    result_label = np.where(result_label==1,255,0)
    result_image = Image.fromarray(np.uint8(result_label))
    result_image.save(output_dir)



## Reconstruction Opening and Closing
def post_process_morphology(image_label, imk, output_dir, kernel_size=(7,7,7,7)):
    # imk = Image.open(label_dir_eliminated)
    # image_label = Image.open(label_dir_ori)
    morphological_size = kernel_size[0]
    im3 = imk.filter(ImageFilter.MinFilter(morphological_size))
    for i in range(5):
        im3 = im3.filter(ImageFilter.MinFilter(morphological_size))
    morphological_size = kernel_size[1]
    im4 = im3.filter(ImageFilter.MaxFilter(morphological_size))
    im4 = ImageChops.multiply(im4, image_label)
    while im4 != im3:
        im3 = im4
        im4 = im3.filter(ImageFilter.MaxFilter(morphological_size))
        im4 = ImageChops.multiply(im4, image_label)
    im2 = im4

    morphological_size = kernel_size[2]
    im3 = im2.filter(ImageFilter.MaxFilter(morphological_size))
    for i in range(5):
        im3 = im3.filter(ImageFilter.MaxFilter(morphological_size))
    morphological_size = kernel_size[3]
    im4 = im3.filter(ImageFilter.MinFilter(morphological_size))
    im4 = ImageChops.add(im4, im2)
    while im4 != im3:
        im3 = im4
        im4 = im3.filter(ImageFilter.MinFilter(morphological_size))
        im4 = ImageChops.add(im4, im2)
    result_label = im4
    result_label.save(output_dir)


def IOU_calculation(ground_truth, prediction):
    true_image = Image.open(ground_truth).convert('L')
    pred_image = Image.open(prediction).convert('L')
    union = ImageChops.add(true_image, pred_image)
    intersection = ImageChops.multiply(true_image,pred_image)
    union = asarray(union)
    intersection = asarray(intersection)
    score = np.sum(intersection) / np.sum(union)
    return score


if __name__ == '__main__':
    if os.path.exists(POST_DIR):
        os.system("rm -rf {}".format(POST_DIR))
        os.system("mkdir {}".format(POST_DIR))

    test_images = natsorted(
            glob(os.path.join(args.input_dir, '*.jpg')) +
            glob(os.path.join(args.input_dir, '*.png'))
        )
    test_labels = natsorted(
            glob(os.path.join(args.label, '*.jpg')) + 
            glob(os.path.join(args.label, '*.png'))
        )
    test_masks = natsorted(
            glob(os.path.join('test/mask', '*.jpg')) + 
            glob(os.path.join('test/mask', '*.png'))
        )
    
    avg_iou = 0
    for idx, (image, label) in enumerate(tqdm(zip(test_images, test_labels))):
        output_name = os.path.join(args.output, 'output{}.jpg'.format(idx+1))
        post_process_sky_elimination(image, label, output_name, args.threshold, args.kernel_size)
    
    # for idx, mask in enumerate(test_masks):
    #     output_name = os.path.join(args.output, 'output{}.jpg'.format(idx+1))
    #     score = IOU_calculation(test_masks[idx], output_name)
    #     avg_iou += score
    #     print('{} IOU Score: {}'.format(idx+1, score))

    # print("AVG IOU: {}".format(avg_iou / 16.0))
        
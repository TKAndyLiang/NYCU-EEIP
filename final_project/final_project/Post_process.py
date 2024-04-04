import sklearn
import numpy as np
import math
import argparse
import skimage
from PIL import Image, ImageFilter, ImageChops
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from numpy import asarray
from matplotlib import pyplot as plt
from skimage.morphology import square

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_image", type=str, default='input.jpg', help="input image file (image)")
parser.add_argument("-l", "--label", type=str, default='label.jpg', help="raw label of the image")
parser.add_argument("-o", "--output", type=str, default="output.jpg", help="output file")
parser.add_argument("-k1", "--kernel_size_1", type=int, nargs=2, default=[5, 5], help="size of structure element in sky elimination")
parser.add_argument("-k2", "--kernel_size_2", type=int, nargs=4, default=[3, 3, 3, 3], help="size of structure element in morphological operations")
parser.add_argument("-t", "--threshold", type=float, default=175, help="lightness threshold of sky elimination")
parser.add_argument("-p", "--process", type=int, choices=[1, 2, 3, 4], default=1, help="which post-process to use (1: Sky elimination, 2:Morphological operations by min/max filter, \
                    3:Morphological operation by square structure element), 4:IOU calculation")
args = parser.parse_args()

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

## sky segmentation and elimination
def post_process_sky_elimination(input_dir, label_dir, output_dir, threshold=175 ,kernel_size=(5, 5)):
    ## Open the input image
    image_in = Image.open(input_dir).convert("RGB")

    ## Pre-process the image by Gaussian blur
    image_in_blurred = image_in.filter(ImageFilter.GaussianBlur(radius=1))

    ## Calculate the lightness of the image and convert it to numpy array
    image_in_blurred_I = image_in_blurred.convert("L")
    image_in_data_I = asarray(image_in_blurred_I).copy()

    ## Thresholding to get high lightness pixels
    image_in_bright = np.where(image_in_data_I > threshold, 255, 0)

    ## Generate seed (marker) of the sky segment on the top side of the image
    image_in_marker = np.zeros(image_in_bright.shape , dtype=np.uint8)
    index = np.where(image_in_bright == 255)
    for x_axis in range(index[0].size):
        if index[0][x_axis] < image_in_data_I.shape[0]/10:
            image_in_marker[index[0][x_axis]][index[1][x_axis]] = 255

    ## Convert array back to image for processing
    image_in_marker = Image.fromarray(image_in_marker)
    image_in_mask = Image.fromarray(np.uint8(image_in_bright))

    ## Reconstruction by dilation to generate sky segment (mask is high lightness pixels)
    morphological_size = kernel_size[0]
    im3 = image_in_marker
    im4 = im3.filter(ImageFilter.MaxFilter(morphological_size))
    im4 = ImageChops.multiply(im4, image_in_mask)
    while im4 != im3:
        im3 = im4
        im4 = im3.filter(ImageFilter.MaxFilter(morphological_size))
        im4 = ImageChops.multiply(im4, image_in_mask)
    image_in_sub = im4

    ## Open the raw label image and eliminates the sky segment in it
    image_label = Image.open(label_dir).convert("L")
    imk = ImageChops.subtract(image_label,image_in_sub)
    
    ## Reconstruction by dilation to regenerate the mis-delected part of the image
    morphological_size = kernel_size[1]
    im3 = imk
    im4 = im3.filter(ImageFilter.MaxFilter(morphological_size))
    im4 = ImageChops.multiply(im4, image_label)
    while im4 != im3:
        im3 = im4
        im4 = im3.filter(ImageFilter.MaxFilter(morphological_size))
        im4 = ImageChops.multiply(im4, image_label)
    result_img = im4

    ## Save the result image
    result_img.save(output_dir)



## Reconstruction Opening and Closing
def post_process_morphology_min_max(label_dir_ori, output_dir, kernel_size=(7,7,7,7)):
    ## Open label image
    image_label = Image.open(label_dir_ori).convert("L")

    ## Reconstruction by opening
    ## Geodesic erosion
    morphological_size = kernel_size[0]
    im3 = image_label.filter(ImageFilter.MinFilter(morphological_size))
    for i in range(5):
        im3 = im3.filter(ImageFilter.MinFilter(morphological_size))

    ## Reconstruction by dilation
    morphological_size = kernel_size[1]
    im4 = im3.filter(ImageFilter.MaxFilter(morphological_size))
    im4 = ImageChops.multiply(im4, image_label)
    while im4 != im3:
        im3 = im4
        im4 = im3.filter(ImageFilter.MaxFilter(morphological_size))
        im4 = ImageChops.multiply(im4, image_label)
    im2 = im4

    ## Reconstruction by closing
    ## Geodesic dilation
    morphological_size = kernel_size[2]
    im3 = im2.filter(ImageFilter.MaxFilter(morphological_size))
    for i in range(5):
        im3 = im3.filter(ImageFilter.MaxFilter(morphological_size))

    ## Reconstruction by erosion
    morphological_size = kernel_size[3]
    im4 = im3.filter(ImageFilter.MinFilter(morphological_size))
    im4 = ImageChops.add(im4, im2)
    while im4 != im3:
        im3 = im4
        im4 = im3.filter(ImageFilter.MinFilter(morphological_size))
        im4 = ImageChops.add(im4, im2)
    result_label = im4

    ## Save result image
    result_label.save(output_dir)

def post_process_morphology_SE(label_dir_ori, output_dir, kernel_size=(3,3,3,3)):
    ## Open label image
    image_label = Image.open(label_dir_ori).convert("L").copy()

    ## Change pixels in the image to binary representation
    image_label = np.where(image_label == 255,1,0)

    ## Reconstruction by opening
    ## Geodesic erosion
    im3 = skimage.morphology.erosion(image_label, square(kernel_size[0]))
    for i in range(5):
        im3 = skimage.morphology.erosion(im3, square(kernel_size[0]))

    ## Reconstruction by dilation
    im4 = skimage.morphology.reconstruction(im3, image_label, method='dilation', footprint=square(kernel_size[1]))
    im2 = im4

    ## Reconstruction by closing
    ## Geodesic dilation
    im3 = skimage.morphology.dilation(im2, square(kernel_size[2]))
    for i in range(5):
        im3 = skimage.morphology.dilation(im3, square(kernel_size[2]))

    ## Reconstruction by erosion
    im4 = skimage.morphology.reconstruction(im3, im2, method='erosion', footprint=square(kernel_size[3]))
    result_label = im4

    ## Change pixel values to 0 or 255 and save image
    result_label = np.where(result_label==1,255,0)
    result_image = Image.fromarray(np.uint8(result_label))
    result_image.save(output_dir)


def IOU_calculation(ground_truth, prediction):
    ## Open label images
    true_image = Image.open(ground_truth)
    pred_image = Image.open(prediction)

    ## Generate intersection and union of true and predicted data
    union = ImageChops.add(true_image, pred_image)
    intersection = ImageChops.multiply(true_image,pred_image)
    union = asarray(union)
    intersection = asarray(intersection)

    ## Calculate IOU (Intersection/Union)
    score = np.float(np.sum(intersection))/np.float(np.sum(union))
    print('IOU Score: {}'.format(score))

if __name__ == '__main__':
    if args.process == 1:
        post_process_sky_elimination(args.input_image, args.label, args.output, args.threshold , args.kernel_size_1)
    elif args.process == 2:
        post_process_morphology_min_max(args.input_image, args.output, args.kernel_size_2)
    elif args.process == 3:
        post_process_morphology_SE(args.input_image, args.output, args.kernel_size_2)
    elif args.process == 4:
        IOU_calculation(args.label, args.input_image)

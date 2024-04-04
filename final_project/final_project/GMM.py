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
parser.add_argument("-o", "--output", type=str, default="output.jpg", help="output file (label)")
parser.add_argument("-p", "--patch_size", type=int, default=15, help="Hue variance calculation patch size")
parser.add_argument("-n", "--num_of_class", type=int, default=10, help="number of clusters in GMM algorithm")
parser.add_argument("-t", "--threshold", type=float, default=10, help="Hue variance threshold of selection")
args = parser.parse_args()

## GMM segmentation
def GMM(input_image_dir, output_image_dir ,patch_size = 15, num_of_group = 10, threshold = 10):
    ## Open image
    image = Image.open(input_image_dir)

    ## Pre-processing (Gaussian blur)
    image_blurred = image.filter(ImageFilter.GaussianBlur(radius=1))

    ## Transform image to HSV domain and convert it to numpy array
    image_blurred_HSV = image_blurred.convert("HSV")
    image_HSV_data = asarray(image_blurred_HSV).copy()

    ## Calculate mean Hue variance
    image_H_variance = np.zeros((image_HSV_data.shape[0],image_HSV_data.shape[1]))
    x = 0
    y = 0
    offset = int((patch_size-1)/2)
    image_H_variance[0:offset, :] = 360
    image_H_variance[:, 0:offset] = 360
    image_H_variance[-1:offset, :] = 360
    image_H_variance[:, -1:offset] = 360
    while x < image_HSV_data.shape[0]-patch_size:
      y = 0
      while y < image_HSV_data.shape[1]-patch_size:
        patch = image_HSV_data[x:x+patch_size, y:y+patch_size, 0]
        image_H_variance[x+offset, y+offset] = np.sum(np.square(patch-np.mean(patch)))/patch_size**2
        y += 1
      x += 1

    ## GMM clustering
    ## Reshape data to fit into GMM algorithm
    im_data = asarray(image_blurred_HSV).copy()
    data = np.reshape(im_data, (-1,3))
    gm = GaussianMixture(n_components=num_of_group, random_state=0).fit_predict(data)
    gm = np.array(gm)
    gm = np.reshape(gm, (im_data.shape[0], im_data.shape[1]))

    ## Select water clusters by mean Hue variance thresholding
    selected = []
    for i in range(num_of_group):
      target = np.where(gm==i,1,0)
      if np.sum(target*image_H_variance)/np.sum(target) < threshold:
        selected.append(i)
    result_image = np.zeros((image_HSV_data.shape[0],image_HSV_data.shape[1]) , dtype=np.uint8)
    for k in selected:
      result_image = result_image + np.where(gm==k,255,0)
    
    ## Convert result array into image and store image
    result_image = Image.fromarray(np.uint8(result_image))
    result_image.save(output_image_dir)


if __name__ == '__main__':
    GMM(args.input_image, args.output, args.patch_size, args.num_of_class, args.threshold)

<Demo method>
For Deep Learning Method 
First set up the enviroment and download needed packages

=> create a new conda env with python version 3.8
=> install the pytorch package in the pytorch_install.txt
=> pip install -r requirement.txt

DL segmentation file system

[final_project]
|----pytorch_install.txt
|----ReadMe.txt
|----[segmentation]
|----|----[saved_images]
|----|----[test]
|----|----|----[image]          -------------------> Inference images
|----|----|----[mask]           -------------------> Ground true
|----|----|----[result]         -------------------> DL output
|----|----|----[post_output]    -------------------> DL Post Processing output
|----|----|----
| ...

How to run:
1. put the images you want to demo into "test/image"
2. modify the test.py of line 26 ~ 32
    configurations 1:<Demo method>
        multi_gpu = True
        test_model = 'Uformer'
        test_dir = 'test/image'
        result_dir = 'test/result'
        check_point = 'checkpoint_best_Uformer_v2.pth.tar'
        cuda_device = 'cuda:0'
    configurations 2:
        multi_gpu = False
        test_model = 'UNET'
        test_dir = 'test/image'
        result_dir = 'test/result'
        check_point = 'checkpoint_maxpool_384_v2.pth.tar'
        cuda_device = 'cuda:0'
3. run ```sh test_script.sh``` on the terminal then get the result in the "test" file


To run KMeans method
1. prepare the files needed in the jupyter notebook
2. run_all

To run GMM segmentation process:
use command: python GMM.py -i <input> -o <output> -p <patch_size> -n <num_of_classes> -t <threshold>
where 	<input> is the input image, 

	<output> is the output image(8bit/pixels label), 

	<patch_size> is the consider patch size of Hue variance(must be odd),

	<num_of_classes> is the number of clusters in the GMM result,

	<threshold> is the mean Hue variance threshold to select possible water body group


To run post-processing:
use command: python Post_process.py -i <input> -l <label> -o <output> -k1 <kernel_size_1> -k2 <kernel_size_2> -t <threshold> -p <process>
where	<input> is the input, need to be process image,

	<label> is the raw label image (only used in sky elimination),

	<output> is the output image,

	<kernel_size_1> are the kernel sizes of morphological operations used in sky elimination (2 arguments: first is for growing sky 	segments and the second is for reconstructing mis-deleted water body),

	<kernel_size_2> are the kernel sizes of morphological operations used in noise spots elimination (4 arguments: kernel sizes of 	geodesic erosion, reconstruction by dilation , geodesic dilation and reconstruction of erosion),

	<threshold> is the threshold of lightness thresholding in sky elimination

	<process> is the process you want to conduct (1 is sky elimination, 2 is noise spots elimination)
1. Compile the eeip_hw1.cpp
=> g++ eeip_hw2.cpp -o eeip_hw2

!!!!!
The input image's name is restricted as input{%d}.bmp, where {} is the number of image.
ex: input1.bmp or input10.bmp ...
!!!!!

2. Demo Low-Luminosity Enhancement
    Method1: PowerLawTransformation
        => ./eeip_hw2 <input image path> -plt 0.5
        output image is stored at the same layer named output{%d}_1.bmp

    Method2: Brightness And Contrast Adjustment
        => ./eeip_hw2 <input image path> -bca 3.0 10.0
        output image is stored at the same layer named output{%d}_2.bmp


3. Demo Sharpness Enhancement
    Method: Laplacian Gaussian Filter (two kinds of filters)
        => ./eeip_hw2 <input image path> -lgf
        output images are stored at the same layer named output{%d}_1.bmp, output{%d}_2.bmp

4. Demo Image Denoising
    Method1: Gaussian Smoothing
        => ./eeip_hw2 <input image path> -gs 3.0 5
        output images are stored at the same layer named output{%d}_1.bmp

    Method2: Bilateral Filtering
        => ./eeip_hw2 <input image path> -bf 3.0 150.0 11
        output images are stored at the same layer named output{%d}_2.bmp

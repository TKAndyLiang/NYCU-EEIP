1. Compile the eeip_hw1.cpp
=> g++ eeip_hw1.cpp -o eeip_hw1

!!!!!
The input image's name is restricted as input{%d}.bmp, where {} is the number of image.
ex: input1.bmp or input10.bmp ...
!!!!!

2. Demo flipping function
=> ./eeip_hw1 <input image path> -hf
output image is stored at the same layer named output{%d}_flip.bmp

3. Demo resolution function
=> ./eeip_hw1 <input image path> -qr
output images are stored at the same layer named output{%d}_1.bmp, output{%d}_2.bmp, output{%d}_3.bmp

4. Demo scaling function
=> ./eeip_hw1 <input image path> -s <ratio>
where the ratio is a float type number
output images are stored at the same layer named output{%d}_up.bmp, output{%d}_down.bmp
1. Compile the eeip_hw3.cpp
=> g++ eeip_hw3.cpp -o eeip_hw3

2. Demo input image 1
    Color constancy:
        => ./eeip_hw3 <input image path> <output image path> -gw 0 0 0
    Enhancement:
        => ./eeip_hw3 <input image path> <output image path> -bca 1.6 10

3. Demo input image 2
    Color constancy:
        => ./eeip_hw3 <input image path> <output image path> -maxrgb 0 0 0
    Enhancement:
        => ./eeip_hw3 <input image path> <output image path> -bca 1.5 10

4. Demo input image 3
    Color constancy:
        => ./eeip_hw3 <input image path> <output image path> -maxrgb 0 0 0
    Enhancement1:
        => ./eeip_hw3 <input image path> <output image path> -plt 0.8
    Enhancement2:
        => ./eeip_hw3 <input image path> <output image path> -bf 5.0 5.0 5
        => ./eeip_hw3 <input image path> <output image path> -bca 1.25 15

5. Demo input image 4
    Color constancy1:
        => ./eeip_hw3 <input image path> <output image path> -gw 0 0 0
    Color constancy2:
        => ./eeip_hw3 <input image path> <output image path> -gw -0.2 -0.18 0

    Enhancement:
        => ./eeip_hw3 <input image path> <output image path> -plt 1.15

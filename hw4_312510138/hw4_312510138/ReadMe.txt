1. Compile the eeip_hw4.cpp
=> g++ eeip_hw4.cpp -o eeip_hw4

2. Demo Deblurring
    input1: Wiener filtering
        => ./eeip_hw4 -db input1.bmp output1.bmp 1.0 17 21 0.07
    
    input2:
        Blind method1
            => ./eeip_hw4 -db input2.bmp output2.bmp 5.0 7 19 0.005
        Blind method2
            => ./eeip_hw4 -db input2.bmp output2.bmp 1.0 17 21 0.07
        Non Blind method (failed)
            => ./eeip_hw4 -ndb input1.bmp input1_ori.bmp input2.bmp output2.bmp 0.1





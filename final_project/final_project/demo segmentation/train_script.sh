python train.py --learning_rate 1e-4 --batch_size 24 \
    --epoch 100 --num_workers 4 --pretrained False \
    --resume False --pretrained_weight checkpoint_latest.pth.tar \
    --optimizer AdamW --model Uformer --gpu '0, 1'
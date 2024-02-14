#/bin/bash

cd ..

python3 cpc.py fit \
    --data /workspaces/betania.silva/data/standartized_balanced/MotionSense \
    --epochs 100 \
    --batch_size 128 \
    --accelerator gpu \
    --devices 1 \
    --load_backbone logs/pretrain/CPC/2024-01-31_21-14-31/checkpoints/last.ckpt \
    --training_mode finetune \
    --window_size 60 \
    --num_classes 6 \
    --encoding_size 150 \
    --conv_model True
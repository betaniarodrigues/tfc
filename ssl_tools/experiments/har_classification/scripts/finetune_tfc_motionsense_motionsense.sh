#/bin/bash

cd ..

./tfc.py fit \
    --data /workspaces/hiaac-m4/ssl_tools/data/standartized_balanced/MotionSense \
    --epochs 100 \
    --batch_size 128 \
    --accelerator gpu \
    --devices 1 \
    --load_backbone logs/pretrain/TFC/2024-01-31_21-07-06/checkpoints/last.ckpt \
    --training_mode finetune \
    --encoding_size 128 \
    --features_as_channels True \
    --length_alignment 60 \
    --in_channels 6 \
    --update_backbone False
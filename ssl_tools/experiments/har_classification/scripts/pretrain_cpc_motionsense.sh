#/bin/bash

cd ..

python3 cpc.py fit \
    --data /workspaces/betania.silva/view_concatenated/MotionSense_cpc \
    --epochs 100 \
    --batch_size 1 \
    --accelerator gpu \
    --devices 1 \
    --training_mode pretrain \
    --checkpoint_metric train_loss \
    --window_size 60 \
    --encoding_size 150 \
    --conv_model True
#/bin/bash

cd ..

for dset in "MotionSense";
do
    python3 cpc.py fit \
        --data /workspaces/betania.silva/view_concatenated/${dset} \
        --epochs 100 \
        --batch_size 1 \
        --accelerator gpu \
        --devices 1 \
        --training_mode pretrain \
        --learning_rate 0.00001 \
        --checkpoint_metric train_loss \
        --window_size 50 \
        --encoding_size  256 \
        --backbone_model conv1D \
        --name cpc_${dset}_pretrain_encoding150
done

    
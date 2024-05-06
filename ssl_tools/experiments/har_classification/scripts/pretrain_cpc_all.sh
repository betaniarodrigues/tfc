#/bin/bash

cd ..

for dset in "uci";
do
    python3 cpc_for_har.py fit \
        --data /workspaces/betania.silva/view_concatenated/${dset} \
        --epochs 5 \
        --batch_size 64 \
        --accelerator gpu \
        --devices 1 \
        --training_mode pretrain \
        --learning_rate 0.001 \
        --checkpoint_metric train_loss \
        --window 50 \
        --overlap 25 \
        --encoding_size  256 \
        --backbone_model conv1D \
        --name cpc_${dset}_pretrain_encoding150
done

    
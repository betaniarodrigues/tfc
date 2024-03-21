#/bin/bash

cd ..

./cpc.py fit \
    --data /workspaces/betania.silva/view_concatenated/KuHar \
    --epochs 10 \
    --batch_size 1 \
    --accelerator gpu \
    --devices 1 \
    --training_mode pretrain \
    --checkpoint_metric train_loss \
    --backbone_model conv1D \
    --window_size 60 \
    --encoding_size 150
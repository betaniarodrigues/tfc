#/bin/bash

cd ..

./cpc.py fit \
    --data /workspaces/betania.silva/data/standartized_balanced/KuHar \
    --epochs 5 \
    --batch_size 2 \
    --accelerator gpu \
    --devices 1 \
    --load_backbone //workspaces/betania.silva/ssl_tools/ssl_tools/experiments/har_classification/logs/pretrain/CPC/2024-03-20_11-44-03/checkpoints/last.ckpt \
    --training_mode finetune \
    --backbone_model conv1D \
    --window_size 60 \
    --num_classes 7 \
    --encoding_size 150 
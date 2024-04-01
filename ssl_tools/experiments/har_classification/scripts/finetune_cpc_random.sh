#/bin/bash

cd ..

for dset in "KuHar" "MotionSense" "UCI" "WISDM" "RealWorld_thigh" "RealWorld_waist";
    do
    python3 cpc.py fit \
        --data /workspaces/betania.silva/data/standartized_balanced/${dset} \
        --epochs 150 \
        --batch_size 256 \
        --accelerator gpu \
        --devices 1 \
        --training_mode finetune \
        --learning_rate 0.0001 \
        --window_size 50 \
        --num_classes 7 \
        --encoding_size 256 \
        --backbone_model conv1D \
        --checkpoint_metric val_loss \
        --name cpc_${dset}_finetune_RANDOM
    done

#/bin/bash

cd ..
for dset in "KuHar" "MotionSense" "UCI" "WISDM" "RealWorld_thigh" "RealWorld_waist"; 
do
    pretrain_path="/workspaces/betania.silva/ssl_tools/ssl_tools/experiments/har_classification/logs/pretrain/cpc_${dset}_pretrain_encoding256";
    for last_pretrain in $(ls ${pretrain_path} | sort -n | tail -n 6);
    do
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
            --load_backbone ${pretrain_path}/${last_pretrain}/checkpoints/last.ckpt \
            --window_size 50 \
            --num_classes 7 \
            --encoding_size 256 \
            --backbone_model conv1D \
            --checkpoint_metric val_loss \
            --name cpc_${last_pretrain}_${dset}_finetune_encoding256
        done
    done
done

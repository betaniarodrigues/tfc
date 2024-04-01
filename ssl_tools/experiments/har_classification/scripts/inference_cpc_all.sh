#/bin/bash


test_path="/workspaces/betania.silva/ssl_tools/ssl_tools/experiments/har_classification/logs/test";

cd ..

for dsetpretrain in "KuHar" "MotionSense" "UCI" "WISDM" "RealWorld_thigh" "RealWorld_waist"; 
do
    pretrain_path="/workspaces/betania.silva/ssl_tools/ssl_tools/experiments/har_classification/logs/finetune/cpc_${dsetpretrain}";
    
    for dsetfinetune in "KuHar" "MotionSense" "UCI" "WISDM" "RealWorld_thigh" "RealWorld_waist"; 
    do
        finetune_path="${pretrain_path}_${dsetfinetune}_finetune_encoding256";

        for last_pretrain in $(ls ${finetune_path} | sort -n | tail -n 36);
        do
            for dset in  "KuHar" "MotionSense" "WISDM" "UCI" "RealWorld_thigh" "RealWorld_waist" "RealWorld"; 
            do 
                python3 ./cpc.py test \
                --data /workspaces/betania.silva/data/standartized_balanced/${dset} \
                --batch_size 256 \
                --accelerator gpu \
                --devices 1 \
                --load ${finetune_path}/${last_pretrain}/checkpoints/last.ckpt \
                --window_size 50 \
                --num_classes 7 \
                --encoding_size 256 \
                --backbone_model conv1D \
                --name EncodingSize256 \
                > ${test_path}/${dset}_${last_pretrain}.log
            done
        done
    done
done
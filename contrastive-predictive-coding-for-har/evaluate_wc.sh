# Call evaluate with classifier with different data_percentage arguments from the list [5,25,100]
for data_percentage in 1 5 10 20 30 40 50 60 70 80 90 100; do
    CUDA_VISIBLE_DEVICES=7 python evaluate_with_classifier.py --data_percentage $data_percentage --dataset RealWorld_raw --saved_model /workspaces/betania.silva/ssl_tools/contrastive-predictive-coding-for-har/models/3_dataset/KuHar_RealWorld_k_28_lr_0.0005_bs_64.pkl
done

#!/bin/bash

# Defina a lista de datasets
datasets=('uci' 'wisdm' 'motionsense' 'kuhar' 'realworld_thigh' 'realworld_waist')

# Defina a lista de modelos
models=('uci_k_28_lr_0.0005_bs_64.pkl' 'wisdm_k_28_lr_0.0005_bs_64.pkl' 'motionsense_k_28_lr_0.0005_bs_64.pkl' 'kuhar_k_28_lr_0.0005_bs_64.pkl' 'realworld_thigh_k_28_lr_0.0005_bs_64.pkl' 'realworld_waist_k_28_lr_0.0005_bs_64.pkl')

# Loop através de cada dataset
for dset in "${datasets[@]}"
do
    # Loop através de cada modelo
    for model in "${models[@]}"
    do
        # Caminho completo para o modelo
        model_path="/workspaces/betania.silva/ssl_tools/contrastive-predictive-coding-for-har/models/May-11-2024/$model"
        
        # Comando Python para executar
        python evaluate_with_classifier.py --dataset $dset --saved_model "$model_path" > "${dset}_${model}_result_freeze.csv"
    done
done

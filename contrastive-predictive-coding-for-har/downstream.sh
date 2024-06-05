# # #!/bin/bash

# # Defina a lista de datasets
# #datasets=('UCI_raw' 'MotionSense_raw' 'KuHar_raw')
# datasets=('MotionSense_raw', 'UCI_raw', 'KuHar_raw', 'RealWorld_raw')

# # Defina a lista de modelos
# models=('KuHar_raw_k_28_lr_0.0005_bs_64.pkl', 'MotionSense_raw_k_28_lr_0.0005_bs_64.pkl', 'UCI_raw_k_28_lr_0.0005_bs_64.pkl')

# # Loop através de cada dataset
# for dset in "${datasets[@]}"
# do
#     # Loop através de cada modelo
#     for model in "${models[@]}"
#     do
#         # Caminho completo para o modelo
#         model_path="/workspaces/betania.silva/ssl_tools/contrastive-predictive-coding-for-har/models/one_dataset/$model"
        
#         # Comando Python para executar
#         python evaluate_with_classifier.py --dataset $dset --saved_model "$model_path" > "${model}_${dset}_freeze.csv"
#     done
# done

#!/bin/bash

# Defina a lista de datasets
#datasets=('UCI_raw')
# Defina a lista de datasets
#datasets=('UCI_raw_g')
datasets=('MotionSense_raw' 'UCI_raw' 'KuHar_raw' 'RealWorld_raw' 'UCI_raw_g' 'UCI_raw_12')
# Defina a lista de modelos
models=('KuHar_MotionSense_RealWorld_k_28_lr_0.0005_bs_64.pkl' 'UCI_MotionSense_RealWorld_k_28_lr_0.0005_bs_64.pkl')

# Diretório onde os resultados serão salvos
output_dir="/workspaces/betania.silva/ssl_tools/contrastive-predictive-coding-for-har/results/freeze"

# Arquivo temporário para armazenar a saída do script Python
temp_output="$output_dir/temp_output.txt"

# Loop através de cada dataset
for dset in "${datasets[@]}"
do
    # Loop através de cada modelo
    for model in "${models[@]}"
    do
        # Caminho completo para o modelo
        model_path="/workspaces/betania.silva/ssl_tools/contrastive-predictive-coding-for-har/models/May-28-2024/$model"
        
        # Nome do arquivo CSV de saída específico para o dataset e modelo atual
        output_csv="$output_dir/${model}_${dset}.csv"
        #temp_output="$output_dir/temp_output_${model}_${dset}.txt"

        # Inicia o arquivo CSV com o cabeçalho apropriado, se o arquivo ainda não existe
        if [ ! -f "$output_csv" ]; then
            echo "Dataset,Pretext Dataset,Phase,Loss,Accuracy,F1 Score,Weighted F1 Score,Best Epoch" > "$output_csv"
        fi

        # Executar o script Python e salvar a saída no arquivo temporário
        python evaluate_with_classifier.py --num_epochs 150 --dataset $dset --saved_model "$model_path" > "$temp_output"

        # Processar a saída com um script Python embutido
        python3 <<EOF
import re
import pandas as pd

# Arquivo temporário com a saída do script
temp_output = "$temp_output"

# Lista para armazenar os dados das métricas da "best epoch"
data = []

# Expressão regular para capturar métricas da fase train, val e test
metric_pattern = re.compile(r"Phase: (train|val|test), loss: ([\\d.]+), accuracy: ([\\d.]+), f1_score: ([\\d.]+), f1_score weighted: ([\\d.]+)")

# Ler o conteúdo do arquivo temporário
with open(temp_output, 'r') as file:
    content = file.read()
    # Encontrar a frase da "best epoch"
    best_epoch_match = re.search(r"The best epoch is (\\d+)", content)
    if best_epoch_match:
        best_epoch = int(best_epoch_match.group(1))
        # Encontrar métricas para todas as fases (train, val, test)
        matches = metric_pattern.findall(content)
        for match in matches:
            phase, loss, accuracy, f1_score, weighted_f1_score = match
            # Adicionar os dados às lista de dicionários
            data.append({
                'Dataset': "$dset",
                'Pretext Dataset': "$model",
                'Phase': phase,
                'Loss': float(loss),
                'Accuracy': float(accuracy),
                'F1 Score': float(f1_score),
                'Weighted F1 Score': float(weighted_f1_score),
                'Best Epoch': best_epoch
            })

# Criar um DataFrame a partir da lista de dicionários
df = pd.DataFrame(data)

# Salvar o DataFrame no arquivo CSV (modo append, para adicionar ao arquivo existente)
with open("$output_csv", 'a') as f:
    df.to_csv(f, header=False, index=False)
EOF

    done
done

# Remover o arquivo temporário
rm "$temp_output"

echo "Processamento concluído. Resultados salvos em arquivos CSV específicos para cada dataset e modelo no diretório $output_dir"

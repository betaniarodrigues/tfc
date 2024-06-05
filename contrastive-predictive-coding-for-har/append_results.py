import os
import pandas as pd
import re  # Importar a biblioteca 're' para usar expressões regulares
import argparse

def append_results(temp_output, output_csv, dset, model):
    # Lista para armazenar os dados das métricas da "best epoch"
    data = []

    # Expressão regular para capturar métricas da fase train, val e test
    metric_pattern = re.compile(r"Phase: (train|val|test), loss: ([\d.]+), accuracy: ([\d.]+), f1_score: ([\d.]+), f1_score weighted: ([\d.]+)")

    # Percorrer todos os arquivos no diretório
    for filename in os.listdir(temp_output):
        if filename.endswith('.csv'):
            csv_path = os.path.join(temp_output, filename)
            with open(csv_path, 'r') as file:
                content = file.read()
                # Encontrar a frase da "best epoch"
                best_epoch_match = re.search(r"The best epoch is (\d+)", content)
                if best_epoch_match:
                    best_epoch = int(best_epoch_match.group(1))
                    # Encontrar métricas para todas as fases (train, val, test)
                    matches = metric_pattern.findall(content)
                    for match in matches:
                        phase, loss, accuracy, f1_score, weighted_f1_score = match
                        # Adicionar os dados às lista de dicionários
                        data.append({
                            'Dataset': dset,
                            'Pretext Dataset': model,
                            'Phase': phase,
                            'Loss': float(loss),
                            'Accuracy': float(accuracy),
                            'F1 Score': float(f1_score),
                            'Weighted F1 Score': float(weighted_f1_score),
                            'Best Epoch': best_epoch
                        })

    # Criar um DataFrame a partir da lista de dicionários
    df = pd.DataFrame(data)

    # Salvar o DataFrame em um arquivo CSV
    df.to_csv(output_csv, index=False)

if __name__ == '__main__':
    # Get the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('temp_output', type=str)
    parser.add_argument('output_csv', type=str)
    parser.add_argument('dset', type=str)
    parser.add_argument('model', type=str)
    args = parser.parse_args()
    append_results(args.temp_output, args.output_csv, args.dset, args.model)
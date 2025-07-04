import os
import shutil
import random
from collections import defaultdict

def preparar_testes_massivos():
    """
    Separa 20% das imagens de cada classe da pasta 'all' 
    para uma nova pasta 'testesmassivos'.
    Execute este script apenas uma vez.
    """
    source_folder = "all"
    dest_folder = "testesmassivos"

    # 1. Verifica se a pasta de origem existe
    if not os.path.isdir(source_folder):
        print(f"Erro: A pasta de origem '{source_folder}' não foi encontrada.")
        return

    # 2. Cria a pasta de destino se ela não existir
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        print(f"Pasta '{dest_folder}' criada com sucesso.")
    else:
        print(f"A pasta '{dest_folder}' já existe. Verifique se a separação já foi feita.")
        # Descomente a linha abaixo se quiser sobrescrever a pasta de testes
        # shutil.rmtree(dest_folder); os.makedirs(dest_folder)

    # 3. Agrupa todas as imagens por valor de moeda
    images_by_coin = defaultdict(list)
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                valor_moeda = int(filename.split('_')[0])
                images_by_coin[valor_moeda].append(filename)
            except (ValueError, IndexError):
                continue

    # 4. Move 20% de cada grupo para a pasta de destino
    print("\nMovendo 20% das imagens para o conjunto de testes massivos...")
    total_moved = 0
    for valor, files in images_by_coin.items():
        # Embaralha a lista para garantir uma seleção aleatória
        random.shuffle(files)
        
        # Calcula o número de arquivos a mover (pelo menos 1, se houver arquivos)
        num_to_move = int(len(files) * 0.2)
        if len(files) > 0 and num_to_move == 0:
            num_to_move = 1
            
        # Seleciona os arquivos a serem movidos
        files_to_move = files[:num_to_move]
        
        print(f"  - Moeda de {valor} centavos: movendo {len(files_to_move)} de {len(files)} imagens.")
        
        # Move os arquivos
        for filename in files_to_move:
            shutil.move(os.path.join(source_folder, filename), os.path.join(dest_folder, filename))
            total_moved += 1
            
    print(f"\nOperação concluída! {total_moved} imagens foram movidas para a pasta '{dest_folder}'.")
    print(f"A pasta '{source_folder}' agora contém as imagens restantes para treinamento.")

if __name__ == '__main__':
    preparar_testes_massivos()
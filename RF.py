import cv2
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# --- 1. Função para extrair características (INCLUI GEOMÉTRICOS E COR ANEL-NUCLEO) ---
def extrair_caracteristicas(contorno, imagem_bgr):
    # Atributos geométricos
    area = cv2.contourArea(contorno)
    perimetro = cv2.arcLength(contorno, True)
    circularidade = 4 * np.pi * area / (perimetro ** 2) if perimetro != 0 else 0
    # Máscara total da moeda
    mascara = np.zeros(imagem_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(mascara, [contorno], -1, 255, -1)
    media_cor_total = cv2.mean(imagem_bgr, mask=mascara)[:3]
    # Cálculo do centro e raio aproximado
    (x, y), raio = cv2.minEnclosingCircle(contorno)
    centro = (int(x), int(y))
    raio = int(raio)
    # Máscara do núcleo (círculo menor, ~60% do raio)
    raio_nucleo = int(raio * 0.6)
    mascara_nucleo = np.zeros_like(mascara)
    cv2.circle(mascara_nucleo, centro, raio_nucleo, 255, -1)
    # Máscara do anel = moeda - núcleo
    mascara_anel = cv2.subtract(mascara, mascara_nucleo)
    # Cor média do núcleo e do anel
    cor_nucleo = cv2.mean(imagem_bgr, mask=mascara_nucleo)[:3]
    cor_anel = cv2.mean(imagem_bgr, mask=mascara_anel)[:3]
    # Diferença absoluta média entre anel e núcleo (em cada canal)
    diff_anel_nucleo = tuple(abs(a - n) for a, n in zip(cor_anel, cor_nucleo))
    diff_anel_nucleo_media = np.mean(diff_anel_nucleo)
    # O vetor de características inclui geométricos, cor total, cor núcleo, cor anel e diferença média
    return [area, perimetro, circularidade, *media_cor_total, *cor_nucleo, *cor_anel, diff_anel_nucleo_media]

# --- 2. Pré-processamento e Segmentação (sem alterações) ---
def segmentar_moedas(imagem):
    if imagem is None: return []
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    borrada = cv2.GaussianBlur(cinza, (5, 5), 1)
    bordas = cv2.Canny(borrada, 50, 150)
    contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contornos

# --- 3. Função para treinar o modelo (APENAS RANDOM FOREST) ---
def treinar_modelo():
    dataset_path = "all"
    X, y = [], []
    print(f"Carregando imagens de treino do diretório: {dataset_path}")
    for filename in os.listdir(dataset_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                valor_moeda = int(filename.split('_')[0])
                img_path = os.path.join(dataset_path, filename)
                imagem_treino = cv2.imread(img_path)
                contornos_treino = segmentar_moedas(imagem_treino)
                if contornos_treino:
                    cnt = max(contornos_treino, key=cv2.contourArea)
                    carac = extrair_caracteristicas(cnt, imagem_treino)
                    X.append(carac)
                    y.append(valor_moeda)
            except Exception as e:
                print(f"Não foi possível processar {filename}: {e}")
    if not X: return None
    print(f"{len(X)} imagens processadas com sucesso para treinamento.")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(np.array(X))
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_scaled, np.array(y))
    print("Modelo Random Forest treinado com sucesso!\n")
    return clf, scaler

# --- 4. Lógica para teste individual (MODIFICADA) ---
def teste_individual(clf, scaler):
    test_folder = "teste"
    try:
        image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(image_files) != 1:
            print(f"Erro: A pasta '{test_folder}' deve conter exatamente uma imagem.")
            return
        
        img_path = os.path.join(test_folder, image_files[0])
        imagem_teste = cv2.imread(img_path)
        if imagem_teste is None:
            print(f"Erro ao carregar a imagem '{img_path}'.")
            return
            
        contornos = segmentar_moedas(imagem_teste)
        if contornos:
            cnt = max(contornos, key=cv2.contourArea)
            # Chamada simplificada
            carac = extrair_caracteristicas(cnt, imagem_teste)
            
            carac_scaled = scaler.transform([carac])
            pred_centavos = clf.predict(carac_scaled)[0]
            pred_reais = pred_centavos / 100.0
            
            print(f"--- Classificação Individual ---")
            print(f"Valor previsto para '{img_path}': R$ {pred_reais:.2f}")
            
            x, y_rect, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(imagem_teste, (x, y_rect), (x + w, y_rect + h), (0, 255, 0), 2)
            cv2.putText(imagem_teste, f"R${pred_reais:.2f}", (x, y_rect - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imshow("Resultado da Classificacao", imagem_teste)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Nenhuma moeda detectada na imagem de teste.")
            
    except FileNotFoundError:
        print(f"Erro: A pasta de teste '{test_folder}' não foi encontrada.")

# --- 5. Lógica para teste em massa (SIMPLIFICADA) ---
def teste_em_massa(clf, scaler):
    test_folder = "testesmassivos"
    y_true, y_pred = [], []
    
    print(f"--- Iniciando Teste em Massa na pasta '{test_folder}' ---")
    try:
        image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print("Nenhuma imagem encontrada na pasta de testes massivos.")
            return

        for filename in image_files:
            valor_real = int(filename.split('_')[0])
            img_path = os.path.join(test_folder, filename)
            imagem_teste = cv2.imread(img_path)
            
            contornos = segmentar_moedas(imagem_teste)
            if contornos:
                cnt = max(contornos, key=cv2.contourArea)
                # Chamada atualizada
                carac = extrair_caracteristicas(cnt, imagem_teste)

                carac_scaled = scaler.transform([carac])
                pred_centavos = clf.predict(carac_scaled)[0]
                y_true.append(valor_real)
                y_pred.append(pred_centavos)
        
        if not y_true:
            print("Nenhuma moeda foi processada com sucesso no teste em massa.")
            return

        print("\n--- Relatório de Classificação ---")
        print(classification_report(y_true, y_pred, zero_division=0))
        
        print("--- Matriz de Confusão ---")
        print("Linhas: Valor Real | Colunas: Valor Previsto")
        labels = sorted(list(set(y_true)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        print(f"Labels: {labels}")
        print(cm)
        print("---------------------------------------------------------------------------")

    except FileNotFoundError:
        print(f"Erro: A pasta de teste '{test_folder}' não foi encontrada.")

# --- 6. Nova função para visualizar os filtros (SIMPLIFICADA) ---
def visualizar_filtros():
    test_folder = "teste"
    print(f"\n--- Visualizando Etapas de Pré-processamento e Atributos ---")
    try:
        image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(image_files) != 1:
            print(f"Erro: A pasta '{test_folder}' deve conter exatamente uma imagem.")
            return
        
        img_path = os.path.join(test_folder, image_files[0])
        imagem_original = cv2.imread(img_path)
        if imagem_original is None:
            print(f"Erro ao carregar a imagem '{img_path}'.")
            return

        print(f"Processando a imagem: {img_path}")
        
        cinza = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2GRAY)
        borrada = cv2.GaussianBlur(cinza, (5, 5), 1)
        bordas = cv2.Canny(borrada, 50, 150)
        contornos, _ = cv2.findContours(bordas.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        imagem_com_contorno = imagem_original.copy()

        if contornos:
            cnt = max(contornos, key=cv2.contourArea)
            cv2.drawContours(imagem_com_contorno, [cnt], -1, (0, 255, 0), 2)
            
            # Chamada atualizada
            carac = extrair_caracteristicas(cnt, imagem_original)
            print("\n--- Atributos Extraídos da Moeda ---")
            print(f"Área: {carac[0]:.2f} | Perímetro: {carac[1]:.2f} | Circularidade: {carac[2]:.4f}")
            print(f"Cor Média Total (B, G, R): ({carac[3]:.2f}, {carac[4]:.2f}, {carac[5]:.2f})")
            print(f"Cor Núcleo (B, G, R): ({carac[6]:.2f}, {carac[7]:.2f}, {carac[8]:.2f})")
            print(f"Cor Anel (B, G, R): ({carac[9]:.2f}, {carac[10]:.2f}, {carac[11]:.2f})")
            print(f"Diferença média Anel-Núcleo: {carac[12]:.2f}")
            print("------------------------------------")

        else:
            print("\nNenhum contorno encontrado na imagem.")

        cv2.imshow("1 - Original", imagem_original)
        cv2.imshow("2 - Cinza", cinza)
        cv2.imshow("3 - Bordas Canny", bordas)
        cv2.imshow("4 - Contorno da Moeda", imagem_com_contorno)
        
        print("\nPressione qualquer tecla para fechar as janelas de visualização.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except FileNotFoundError:
        print(f"Erro: A pasta de teste '{test_folder}' não foi encontrada.")


# --- Função principal com menu ---
def main():
    clf, scaler = treinar_modelo()
    if clf is None:
        print("Falha no treinamento. Encerrando o programa.")
        return
    while True:
        print("\nEscolha uma opção (main2.py - Random Forest):")
        print("1 - Classificar uma única imagem da pasta 'teste'")
        print("2 - Realizar teste em massa com a pasta 'testesmassivos' e gerar relatório")
        print("3 - Visualizar etapas de pré-processamento da imagem de teste")
        print("4 - Sair")
        choice = input("Digite sua escolha (1, 2, 3 ou 4): ")
        if choice == '1':
            teste_individual(clf, scaler)
        elif choice == '2':
            teste_em_massa(clf, scaler)
        elif choice == '3':
            visualizar_filtros()
        elif choice == '4':
            print("Encerrando o programa.")
            break
        else:
            print("Opção inválida. Por favor, tente novamente.")

if __name__ == '__main__':
    main()

import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model # type: ignore

## carrega o modelo treinado
model = load_model('modelo_letras.h5')

## caminho da pasta com as imagens
pasta = 'imagens_teste'

## lista os arquivos da pasta em ordem alfabetica)
arquivos = sorted(os.listdir(pasta))

## contadores para acertos e total
total = 0
acertos = 0

## testa cada imagem da pasta
for nome_arquivo in arquivos:
    if nome_arquivo.endswith('.png'):
        caminho = os.path.join(pasta, nome_arquivo)
        img = Image.open(caminho).convert('L') ## converte pra tons de cinza
        img = 255 - np.array(img) ## inverte as cores
        img = img / 255.0 ## normaliza
        img = img.reshape(1, 28, 28) ## ajusta formato para o modelo

        ## faz a previsão
        previsao = model.predict(img, verbose=0)
        indice = np.argmax(previsao)
        letra_prevista = chr(ord('A') + indice)

        ## extrai a letra real do nome do arquivo, por exemplo "letra_A.png"
        letra_real = nome_arquivo.split('_')[1].split('.')[0]

        ## mostra o resultado
        print(f"Imagem: {nome_arquivo} -> Previsão: {letra_prevista}")

        ## atualiza contadores
        total += 1
        if letra_real.upper() == letra_prevista:
            acertos += 1

## calcula percentual de acerto
percentual = (acertos / total) * 100 if total > 0 else 0

print(f"\nTotal de imagens testadas: {total}")
print(f"Número de acertos: {acertos}")
print(f"Percentual total de acerto: {percentual:.2f}%")

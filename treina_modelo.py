import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

## 1 - carregando base de imagens emnist
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/letters',
    split=['train', 'test'], ## divide dados entre treinamento e teste
    shuffle_files=True, ## embaralha os arquivos para melhorar o aprendizado
    as_supervised=True, ## faz as imagens virem no formato [imagem, classe]
    with_info=True ## retorna informações da imagem como tamanho
)

## 2 - pré processando os dados
def normaliza(x, y):
    x = tf.cast(x, tf.float32) / 255.0 ## converte os pixels da imagem para float
    ## é dividido por 255 para os valores dos pixels irem de 0.0 a 1.0
    y = tf.cast(y, tf.int32) - 1 ## as classes vem de A (1) até Z (26)
    ## subtraimos 1 para alinhar ao python onde contagem começa em 0
    return x, y ## retorna a imagem e a classe ajustadas e normalizadas

ds_train = ds_train.map(normaliza).batch(128).prefetch(1) 
ds_test = ds_test.map(normaliza).batch(128).prefetch(1)
## map(normaliza) aplica a função normaliza a cada item do dataset
## batch(128) agrupa 128 por vez, para acelerar o treinamento
## prefettch(1) deixa o proximo lote preparado na memória, agilizando o desempenho


###########################
########## criando o modelo
###########################


## 3 - criando rede neural
model = Sequential([ ## cria um modelo que as camadas são empilhadas uma após outra
    Flatten(input_shape=(28, 28)), ## transforma imagem 28x28 em vetor
    Dense(128, activation='relu'), ## camada oculta com 128 neuronios para encontrar os padrões. relu ativa só neuronios uteis
    Dense(26, activation='softmax') ## camda final que devolve 26 probabilidades (A-Z). a maior delas é a letra prevista
])

## 4 - compilando o modelo
model.compile(
    optimizer='adam', ## otimizador inteligente que ajusta os pesos do modelo a cada tentativa
    loss='sparse_categorical_crossentropy', ## mede o erro do modelo durante o treino
    metrics=['accuracy'] ## mostra a accuracy de acertos em %
)

## 5 - treinando o modelo
model.fit( ## começa o treinamento
    ds_train, ## vai aprender cm os dados do ds_train
    epochs=5, ## o modelo vai olhar todos os dados 5 vezes. a cada geração ele melhora
    validation_data=ds_test ## a cada geração o modelo é testado com os dados para ver como esta indo
)

## 6 - salvando o modelo
model.save('modelo_letras.h5') ## cerebro treinado com tudo que o modelo aprendeu
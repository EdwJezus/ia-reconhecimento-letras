# Reconhecimento de Letras com Keras (EMNIST)

Este projeto implementa um sistema de reconhecimento automático de letras (A-Z), utilizando aprendizado de máquina com Keras e TensorFlow. O modelo foi treinado com o dataset **EMNIST Letters** e é capaz de identificar letras a partir de imagens de entrada em tons de cinza.

## 👨‍💻 Desenvolvido por
- Eduardo Jesus Antonio Pereira Peres  
- Rebeca de Oliveira Maier  

---

## 📁 Estrutura do Projeto

```
.
├── treina_modelo.py          # Script que treina o modelo
├── modelo_letras.h5          # Modelo treinado salvo (formato H5)
├── reconhece_imagem.py       # Script de reconhecimento
├── imagens_teste/            # Imagens para teste do modelo (letra_A.png, ..., letra_Z.png)
├── historico_treino.json     # Histórico de treino (acurácia e perda por época)
├── grafico_resultado.py      # Gera gráfico com base no histórico
├── requirements.txt          # Bibliotecas necessárias
└── README.md                 # Este documento
```

---

## 📦 Instalação

1. **Clone este repositório**:
   ```bash
   git clone https://github.com/EdwJezus/ia-reconhecimento-letras.git
   cd ia-reconhecimento-letras
   ```

2. **Instale os requisitos**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧠 Treinamento do Modelo

> O treinamento **já foi realizado**, mas você pode refazer com:

```bash
python treina_modelo.py
```

O script `treina_modelo.py` irá:
- Carregar o dataset EMNIST Letters via `tensorflow_datasets`
- Pré-processar os dados
- Construir uma rede MLP simples (camadas densas)
- Treinar por 5 épocas
- Salvar o modelo em `modelo_letras.h5`
- Salvar o histórico em `historico_treino.json`

---

## 🔎 Reconhecimento de Imagens

Para testar o modelo com as imagens da pasta `imagens_teste/`:

```bash
python reconhece_imagem.py
```

O terminal exibirá a previsão para cada imagem e o percentual de acerto final:

```
Imagem: letra_A.png -> Previsão: A
Imagem: letra_B.png -> Previsão: B
...
Total de imagens testadas: 26
Número de acertos: 23
Percentual total de acerto: 88.46%
```

---

## 📊 Visualização do Treinamento

Você pode gerar um gráfico com as curvas de acurácia e perda usando:

```bash
python grafico_treino.py
```

Isso exibirá um gráfico com o desempenho do modelo em treino e validação.

---

## 🧪 Formato das Imagens de Teste

As imagens devem seguir o seguinte padrão:
- Formato: `.png`
- Tamanho: 28x28 pixels
- Cor: **letra preta** sobre **fundo branco**
- Nome: `letra_A.png`, `letra_B.png`, ..., `letra_Z.png`

As imagens em `imagens_teste/` foram extraídas diretamente do EMNIST para garantir compatibilidade.

---

## 📈 Resultados

- **Acurácia de validação:** ~88% (com imagens EMNIST)
- **Modelo:** MLP com 2 camadas densas (`relu` + `softmax`)
- **Treinamento:** 5 épocas
- **Dataset:** EMNIST Letters (via `tensorflow_datasets`)

---

## 📚 Referências

- [EMNIST Dataset – NIST](https://www.nist.gov/itl/products-and-services/emnist-dataset)
- [Keras Documentation](https://keras.io/)
- [TensorFlow Datasets](https://www.tensorflow.org/datasets)
- [Professor Filipo – Códigos de Aula](https://github.com/ProfessorFilipo/PythonAI/tree/main/DeepLearning)

---

## 📝 Observações Finais

- O projeto foi adaptado para usar imagens do próprio EMNIST na fase de teste, pois imagens externas (desenhadas manualmente) geravam baixo desempenho, mesmo com modelos mais complexos.
- Após testes com CNNs, mais épocas e alterações de arquitetura, a versão atual (MLP simples) demonstrou excelente desempenho com EMNIST e foi mantida por simplicidade e eficiência.

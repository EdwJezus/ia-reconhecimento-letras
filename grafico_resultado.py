import json
import matplotlib.pyplot as plt

## carrega o historico salvo
with open('historico_treino.json', 'r') as f:
    history = json.load(f)

## accuracy
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Treino')
plt.plot(history['val_accuracy'], label='Teste')
plt.title('Acurácia por época')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True)

## perda (loss)
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Treino')
plt.plot(history['val_loss'], label='Teste')
plt.title('Perda (Loss) por época')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
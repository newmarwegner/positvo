# Atividade em aula - AT09
# Autor: Newmar Wegner
# Date: 10/04/2021

"""
ATIVIDADE
1. Usando o modelo salvo para fertility, criar um progama para predições
"""
from pickle import load

# Function to load saved model
def load_model():
    return load(open('../datasets/model.pkl', 'rb'))


if __name__ == '__main__':
    rf_fertility = load_model()
    print(rf_fertility.predict([[-0.33, 0.5, 1, 1, 0, -1, 0.8, 0, 0.88]]))

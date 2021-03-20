# Atividade em aula - AT07
# Autor: Newmar Wegner
# Date: 20/03/2021

'''
ATIVIDADE
1. Gerar um predict a partir do modelo salvo para identificar a qual cluster ele pertence
'''
import numpy as np
from pickle import load

to_predict = np.array([[-0.33,0.69,0,1,1,0,0.8,0,0.88]])
model = load(open('results/modelclusterfertility.pkl', 'rb'))
print(model.predict(to_predict))

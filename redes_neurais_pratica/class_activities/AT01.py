# Atividade em aula - AT01
# Autor: Newmar Wegner
# Date: 07/05/2021


# preparar dados

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

X_np = torch.range(1, 10)
y_np = X_np ** 2

X = X_np.view(X_np.shape[0], 1)
y = y_np.view(y_np.shape[0], 1)

# Criar modelo
# n_sample, n_features = X.shape
model = nn.Sequential(nn.Linear(1, 10),
                      nn.ReLU(),
                      nn.Linear(10, 1))

# Função de erro
criterion = nn.L1Loss()
# Otimizador
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

# Treinar modelo
epochs = 20000

for e in range(epochs):
    # forward and loss
    y_predict = model(X)
    loss = criterion(y_predict, y)
    # backward
    loss.backward()
    # weights update and zero grad
    optimizer.step()
    optimizer.zero_grad()
    if (e + 1) % 10 == 0: print(f'epoch: {e + 1} loss: {loss.item():.4f}')

predicted = model(X).detach().numpy()
plt.plot(X_np, y_np, 'ro')
plt.plot(X_np, predicted, 'b')
plt.show()

# gerar log para utilizar no tensorboard
writer = SummaryWriter('../results/')
writer.add_graph(model, X)
writer.close()

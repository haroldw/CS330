import torch
import tqdm
import matplotlib.pyplot as plt
import numpy as np

from util import generate_data_categorical, plot_loss
from models import StructuralModel

N = 10
model = StructuralModel(N, dtype=torch.float64)

num_episodes = 50
batch_size = 100  # 1
num_test = 5000
num_training = 10 # 10  # 100
num_transfers = 10  # 100

optimizer = torch.optim.SGD(model.modules_parameters(), lr=1.0)

losses = np.zeros((model.n_models, num_training, num_transfers, num_episodes))

for k in tqdm.trange(num_training):
    pi_A_1 = np.random.dirichlet(np.ones(N))
    pi_B_A = np.random.dirichlet(np.ones(N), size=N)
    pi_C_B = np.random.dirichlet(np.ones(N), size=N)

    for w in range(5000):
        x_transfer = generate_data_categorical(batch_size, pi_A_1, pi_B_A, pi_C_B)
        model.zero_grad()
        loss = sum([-torch.mean(l) for l in model.compute_losses(x_transfer)])
        loss.backward()
        optimizer.step()
    model.save_weight_snapshot()

    for j in tqdm.trange(num_transfers, leave=False):
        model.load_weight_snapshot()
        pi_A_2 = np.random.dirichlet(np.ones(N))
        x_val = generate_data_categorical(num_test, pi_A_2, pi_B_A, pi_C_B)
        for i in range(num_episodes):
            x_transfer = generate_data_categorical(batch_size, pi_A_2, pi_B_A, pi_C_B)
            model.zero_grad()
            loss = sum([-torch.mean(l) for l in model.compute_losses(x_transfer)])

            with torch.no_grad():
                val_loss = [-torch.mean(l) for l in model.compute_losses(x_val)]
            losses[:, k, j, i] = val_loss

            loss.backward()
            optimizer.step()

plot_loss(losses, model, "three_variables_loss_comparison.png")

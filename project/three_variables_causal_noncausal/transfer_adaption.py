import torch
import tqdm
import matplotlib.pyplot as plt
import numpy as np

from util import generate_data_categorical, plot_loss
from models import StructuralModel

N = 10
model = StructuralModel(N, dtype=torch.float64)

num_episodes = 1000
batch_size = 100  # 1
num_test = 10000
num_training = 10  # 100
num_transfers = 10  # 100

optimizer = torch.optim.SGD(model.modules_parameters(), lr=1.0)

losses = np.zeros((2, num_training, num_transfers, num_episodes))

for k in tqdm.trange(num_training):
    pi_A_1 = np.random.dirichlet(np.ones(N))
    pi_B_A = np.random.dirichlet(np.ones(N), size=N)
    pi_C_B = np.random.dirichlet(np.ones(N), size=N)

    for w in range(1000):
        x_transfer = generate_data_categorical(batch_size, pi_A_1, pi_B_A, pi_C_B)
        model.zero_grad()
        loss_A_B = -torch.mean(model.model_causal(x_transfer))
        loss_B_A = -torch.mean(model.model_noncausal(x_transfer))
        loss_C_B = -torch.mean(model.model_noncausal(x_transfer))
        loss = loss_A_B + loss_B_A + loss_C_B
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
            loss_causal = -torch.mean(model.model_causal(x_transfer))
            loss_noncausal = -torch.mean(model.model_noncausal(x_transfer))
            loss = loss_causal + loss_noncausal

            with torch.no_grad():
                val_loss_causal = -torch.mean(model.model_causal(x_val))
                val_loss_noncausal = -torch.mean(model.model_noncausal(x_val))
            losses[:, k, j, i] = [val_loss_causal.item(), val_loss_noncausal.item()]

            loss.backward()
            optimizer.step()

plot_loss(losses, "three_variables_loss_comparison.png")

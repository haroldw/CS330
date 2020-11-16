import torch
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from util import generate_data_categorical, plot_loss
from models import StructuralModel

N = 5
num_episodes = 200
batch_size = 50 # 1
num_test = batch_size
num_training = 1 # 100
num_transfers = 10 # 100

model = StructuralModel(N, batch_size, dtype=torch.float64)
model1 = StructuralModel(N, batch_size, dtype=torch.float64)
optimizer = torch.optim.SGD(model.modules_parameters(), lr=0.1)
optimizer1 = torch.optim.SGD(model1.modules_parameters(), lr=0.1)
losses = np.zeros((2, num_training, num_transfers, num_episodes))

trainingLoss = {
    'marginalConditional': [],
    'joint': [],
    'nonCausal': [],
}
for k in tqdm.trange(num_training):
    pi_A_1 = np.random.dirichlet(np.ones(N))
    pi_B_A = np.random.dirichlet(np.ones(N), size=N)
    pi_all = np.multiply(pi_B_A, pi_A_1)
    for w in range(int(1E3)):
        x_transfer = generate_data_categorical(batch_size, pi_A_1, pi_B_A)
        model.zero_grad()

        causalPredA, causalPredAB, causalJoint = model.model_causal(x_transfer)
        # causalLoss = torch.mean(torch.norm(causalPredA-torch.tensor(pi_A_1), p=None) +
        #                   torch.norm(causalPredAB-torch.tensor(pi_B_A), p=None))
        causalLoss = torch.mean(torch.norm(causalJoint-torch.tensor(pi_all), p=None))

        _, _, causalJoint = model.model_causal(x_transfer)
        causalLoss1 = torch.mean(torch.norm(causalJoint-torch.tensor(pi_all), p=None))

        nonCausalPred = model.model_noncausal(x_transfer)
        noncausalLoss = torch.mean(torch.norm(nonCausalPred-torch.tensor(pi_all),
                                              p=None))
        if w % 100 == 0:
            print(f'Iteration {w}, Noncausal loss is {noncausalLoss.item():.2e},' +
                  f'causal Loss is {causalLoss.item():.2e},' +
                  f'causal loss1 is {causalLoss1.item():.2e}' )
            trainingLoss['marginalConditional'].append(causalLoss.item())
            trainingLoss['joint'].append(causalLoss1.item())
            trainingLoss['nonCausal'].append(noncausalLoss.item())
        causalLoss.backward()
        causalLoss1.backward
        noncausalLoss.backward()
        optimizer.step()
        optimizer1.step()

    torch.save(model.model_causal, './causalModel')
    torch.save(model.model_noncausal, './noncausalModel')

    for j in tqdm.trange(num_transfers, leave=False):
        model.model_causal = torch.load('./causalModel')
        model.model_noncausal = torch.load('./noncausalModel')
        optimizer = torch.optim.SGD(model.modules_parameters(), lr=0.1)
        pi_A_2 = np.random.dirichlet(np.ones(N))
        pi_all = np.multiply(pi_B_A, pi_A_2)
        x_val = []
        for h in range(50):
            x_val.append(generate_data_categorical(num_test, pi_A_2, pi_B_A))

        for i in range(num_episodes):
            x_transfer = generate_data_categorical(batch_size, pi_A_2, pi_B_A)
            model.zero_grad()

            causalPredA, causalPredAB, causalPred = model.model_causal(x_transfer)
            causalLoss = torch.mean(torch.norm(causalPred-torch.tensor(pi_all) +
                                               causalPredA-torch.tensor(pi_A_2),
                                               p=None))
            nonCausalPred = model.model_noncausal(x_transfer)
            noncausalLoss = torch.mean(torch.norm(nonCausalPred-torch.tensor(pi_all),
                                              p=None))
            
            with torch.no_grad():
                valLossCausal = 0
                valLossNoncausal = 0
                for h in range(50):
                    _, causalPredAB, causalPred = model.model_causal(x_val[h])
                    valLossCausal += torch.mean(
                        torch.norm(causalPred-torch.tensor(pi_all), p=None)).item()
                    # valLossCausal += torch.mean(torch.norm(causalPredAB-torch.tensor(pi_all), p=None)).item()
                    nonCausalPred = model.model_noncausal(x_val[h])
                    valLossNoncausal += torch.mean(torch.norm(nonCausalPred-torch.tensor(pi_all),
                                            p=None)).item()
            '''
            loss[num_mode, num_training, num_transfer, num_episodes]
            '''
            losses[:, k, j, i] = [valLossCausal, valLossNoncausal]

            causalLoss.backward()
            noncausalLoss.backward()
            optimizer.step()

plot_loss(losses, "NN_loss_comparison.png")
import torch
import tqdm
import matplotlib.pyplot as plt
import numpy as np

from util import generate_data_categorical, plot_loss
from models import StructuralModel

N = 10
model = StructuralModel(N, dtype=torch.float64)

num_episodes = 1000
batch_size = 100 # 1
num_test = 10000
num_training = 1 # 100
num_transfers = 10 # 100

optimizer = torch.optim.SGD(model.modules_parameters(), lr=1.)

losses = np.zeros((2, num_training, num_transfers, num_episodes))


'''
  1. Initalize causal and non causal
  2. Randomlize P(A) and P(B|A) 
  3. Generate data with P(A) and P(B|A)
      generate_data_categorical(num_test, pi_A_1, pi_B_A)
  4. Train causal and non-causal model
  5. Update P(A) to P2(A)
  6. Regenerate data with P2(A) and P(B|A)
  7. Meta learn with new data
'''

'''
  1. Initalize P_A_B and P_B_A
  2. Randomlize P(A) and P(B|A) 
  3. Calculate P(B) and P(A|B)
  4. Set ground truth P_A_B ( p(A), P(B|A))
  5. Set ground truth P_B_A ( P(B), P(A|B))
  6. Update P(A) to P2(A)
  6. Regenerate data with P2(A) and P(B|A)
  7. Meta learn with new data
'''

'''
  1. Initalize causal and non-causal
    - Causal P(A), P(B|A), P(C|B)
    - Non-causal 
  2. Randomlize P(A), P(B|A), P(C|B)
  3. Generate data A, B, C
  4. Train causal and non-causal model
  5. Update P(A) to P2(A)
  6. Regenerate data with P2(A) and P(B|A)
  7. Meta learn with new data
'''


for k in tqdm.trange(num_training):
    pi_A_1 = np.random.dirichlet(np.ones(N))
    pi_B_A = np.random.dirichlet(np.ones(N), size=N)

    for w in range(1000):
        x_transfer = generate_data_categorical(batch_size, pi_A_1, pi_B_A)
        model.zero_grad()
        loss_A_B = -torch.mean(model.model_causal(x_transfer))
        loss_B_A = -torch.mean(model.model_noncausal(x_transfer))
        loss = loss_A_B + loss_B_A
        loss.backward()
        optimizer.step()
    model.save_weight_snapshot()

    for j in tqdm.trange(num_transfers, leave=False):
        model.load_weight_snapshot()
        pi_A_2 = np.random.dirichlet(np.ones(N))
        x_val = generate_data_categorical(num_test, pi_A_2, pi_B_A)
        for i in range(num_episodes):
            x_transfer = generate_data_categorical(batch_size, pi_A_2, pi_B_A)
            model.zero_grad()
            loss_A_B = -torch.mean(model.model_causal(x_transfer))
            loss_B_A = -torch.mean(model.model_noncausal(x_transfer))
            loss = loss_A_B + loss_B_A
            
            with torch.no_grad():
                val_loss_A_B = -torch.mean(model.model_causal(x_val))
                val_loss_B_A = -torch.mean(model.model_noncausal(x_val))
            '''
            loss[num_mode, num_training, num_transfer, num_episodes]
            '''
            losses[:, k, j, i] = [val_loss_A_B.item(), val_loss_B_A.item()]

            loss.backward()
            optimizer.step()

plot_loss(losses)
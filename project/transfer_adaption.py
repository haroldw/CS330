import torch
import tqdm
import matplotlib.pyplot as plt
import numpy as np

from util import generate_data_categorical
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
    for j in tqdm.trange(num_transfers, leave=False):
        model.initialize_weights()
        pi_A_2 = np.random.dirichlet(np.ones(N))
        x_val = torch.from_numpy(generate_data_categorical(num_test, pi_A_2, pi_B_A))
        for i in range(num_episodes):
            x_transfer = torch.from_numpy(generate_data_categorical(batch_size, pi_A_2, pi_B_A))
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

flat_losses = -losses.reshape((2, -1, num_episodes))
losses_25, losses_50, losses_75 = np.percentile(flat_losses, (25, 50, 75), axis=1)

plt.figure(figsize=(9, 5))

ax = plt.subplot(1, 1, 1)
ax.plot(losses_50[0], color='C0', label=r'Causal', lw=2)
ax.fill_between(np.arange(num_episodes), losses_25[0], losses_75[0], color='C0', alpha=0.2)
ax.plot(losses_50[1], color='C3', label=r'Non Causal', lw=2)
ax.fill_between(np.arange(num_episodes), losses_25[1], losses_75[1], color='C3', alpha=0.2)
ax.set_xlim([0, 50])
ax.tick_params(axis='both', which='major', labelsize=13)
ax.legend(loc=4, prop={'size': 13})
ax.set_xlabel('Number of examples', fontsize=14)
ax.set_ylabel(r'$\log P(D\mid \cdot \rightarrow \cdot)$', fontsize=14)

plt.show()

import torch
import torch.nn.functional as F
import numpy as np

def generate_data_categorical(num_samples, pi_A, pi_B_A):
    """Sample data using ancestral sampling
    
    x_A ~ Categorical(pi_A)
    x_B ~ Categorical(pi_B_A[x_A])
    """
    N = pi_A.shape[0]
    r = np.arange(N)
    
    x_A = np.dot(np.random.multinomial(1, pi_A, size=num_samples), r)
    x_Bs = np.zeros((num_samples, N), dtype=np.int64)
    for i in range(num_samples):
        x_Bs[i] = np.random.multinomial(1, pi_B_A[x_A[i]], size=1)
    x_B = np.dot(x_Bs, r)
    
    return np.vstack((x_A, x_B)).T.astype(np.int64)

def logsumexp(a, b):
    min_, max_ = torch.min(a, b), torch.max(a, b)
    return max_ + F.softplus(min_ - max_)
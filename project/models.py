import numpy as np
import torch
import torch.nn as nn

from itertools import chain
from util import logsumexp

class Marginal(nn.Module):
  def __init__(self, N, dtype=None):
      super(Marginal, self).__init__()
      self.N = N
      self.w = nn.Parameter(torch.zeros(N, dtype=dtype))

  def forward(self, inputs):
      '''
      w is the log probability of each input value
      cste should be 0
      '''
      cste = torch.logsumexp(self.w, dim=0)
      return self.w[inputs.squeeze(1)] - cste

class Conditional(nn.Module):
    def __init__(self, N, dtype=None):
        super(Conditional, self).__init__()
        self.N = N
        self.w = nn.Parameter(torch.zeros((N, N), dtype=dtype))
    
    def forward(self, inputs, conds):
        conds_ = conds.squeeze(1)
        cste = torch.logsumexp(self.w[conds_], dim=1)
        return self.w[conds_, inputs.squeeze(1)] - cste

class ModelNonCausal(nn.Module):
    '''
    The non-causal model captures P(A,B)
    '''
    def __init__(self, N, dtype=None):
        super(ModelNonCausal, self).__init__()
        self.N = N
        self.w = nn.Parameter(torch.zeros((N, N), dtype=dtype))

    def forward(self, inputs):
        # We need to normalize the weight so that it adds up to 1
        cste = torch.logsumexp(torch.logsumexp(self.w, dim=0), dim=0)
        return self.w[inputs[:,0], inputs[:,1]] - cste

    def set_ground_truth(self, pi_A, pi_B_A):
        # Initializing the model parameter to some random value
        pi_A_AND_B = np.random.dirichlet(np.ones(self.N), size=self.N)
        pi_A_AND_B = torch.from_numpy(pi_A_AND_B)        
        self.w.data = torch.log(pi_A_AND_B)

class ModelCausal(nn.Module):
    '''
    The causal model captures P(B|A)P(A)
    '''
    def __init__(self, N, dtype=None):
        super(ModelCausal, self).__init__()
        self.N = N
        self.p_A = Marginal(N, dtype=dtype)
        self.p_B_A = Conditional(N, dtype=dtype)

    def forward(self, inputs):
        inputs_A, inputs_B = torch.split(inputs, 1, dim=1)
        return self.p_A(inputs_A) + self.p_B_A(inputs_B, inputs_A)

    def set_maximum_likelihood(self, inputs):
        inputs_A, inputs_B = np.split(inputs.numpy(), 2, axis=1)
        num_samples = inputs_A.shape[0]
        pi_A = np.zeros((self.N,), dtype=np.float64)
        pi_B_A = np.zeros((self.N, self.N), dtype=np.float64)
        
        # Empirical counts for p(A)
        for i in range(num_samples):
            pi_A[inputs_A[i, 0]] += 1
        pi_A /= float(num_samples)
        assert np.isclose(np.sum(pi_A, axis=0), 1.)
        
        # Empirical counts for p(B | A)
        for i in range(num_samples):
            pi_B_A[inputs_A[i, 0], inputs_B[i, 0]] += 1
        pi_B_A /= np.maximum(np.sum(pi_B_A, axis=1, keepdims=True), 1.)
        sum_pi_B_A = np.sum(pi_B_A, axis=1)
        assert np.allclose(sum_pi_B_A[sum_pi_B_A > 0], 1.)

        return self.set_ground_truth(pi_A, pi_B_A)

    def set_ground_truth(self, pi_A, pi_B_A):
        # Initializing the model parameter to some random value
        pi_A = np.random.dirichlet(np.ones(self.N))
        pi_B_A = np.random.dirichlet(np.ones(self.N), size=self.N)
        pi_A_th = pi_A
        if isinstance(pi_A_th, np.ndarray):
          pi_A_th = torch.from_numpy(pi_A_th)

        pi_B_A_th = pi_B_A
        if isinstance(pi_B_A_th, np.ndarray):
          pi_B_A_th = torch.from_numpy(pi_B_A_th)
        
        self.p_A.w.data = torch.log(pi_A_th)
        self.p_B_A.w.data = torch.log(pi_B_A_th)

class StructuralModel(nn.Module):
    def __init__(self, N, dtype=None):
        super(StructuralModel, self).__init__()
        self.model_causal = ModelCausal(N, dtype=dtype)
        self.model_noncausal = ModelNonCausal(N, dtype=dtype)
        self.w = nn.Parameter(torch.tensor(0., dtype=dtype))
    
    def set_ground_truth(self, pi_A, pi_B_A):
        self.model_causal.set_ground_truth(pi_A, pi_B_A)
        
        # Calculate P(B) and P(A|B)
        pi_A_th = torch.from_numpy(pi_A)
        pi_B_A_th = torch.from_numpy(pi_B_A)
        log_joint = torch.log(pi_A_th.unsqueeze(1)) + torch.log(pi_B_A_th)
        log_p_B = torch.logsumexp(log_joint, dim=0)
        pi_B = torch.exp(log_p_B)
        pi_A_B = torch.exp(log_joint.t() - log_p_B.unsqueeze(1))

        self.model_noncausal.set_ground_truth(pi_B, pi_A_B)
    
    def set_maximum_likelihood(self, inputs):
        self.model_causal.set_maximum_likelihood(inputs)
        self.model_noncausal.set_maximum_likelihood(inputs)

    def forward(self, inputs):
        return self.online_loglikelihood(self.model_causal(inputs), self.model_noncausal(inputs))

    def online_loglikelihood(self, logl_A_B, logl_B_A):
        n = logl_A_B.size(0)
        log_alpha, log_1_m_alpha = F.logsigmoid(self.w), F.logsigmoid(-self.w)

        return logsumexp(log_alpha + torch.sum(logl_A_B),
            log_1_m_alpha + torch.sum(logl_B_A))# / float(n)

    def modules_parameters(self):
        return chain(self.model_causal.parameters(), self.model_noncausal.parameters())

    def structural_parameters(self):
        return [self.w]

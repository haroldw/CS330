import numpy as np
import torch
import torch.nn as nn

from itertools import chain
from numpy.core.defchararray import mod
from torch.nn.modules.module import Module
from util import logsumexp

class NNModule(nn.Module):
    def __init__(self, N, dtype=None):
        super(NNModule, self).__init__()
        self.N = N

    def save_weight_snapshot(self):
        # w_ is snapshot of w, used to recover the NN weights
        self.w_ = self.w

    def load_weight_snapshot(self):
        self.w = self.w_

class Marginal(NNModule):
    def __init__(self, N, dtype=None):
        super(Marginal, self).__init__(N)
        self.w = nn.Parameter(torch.zeros(N, dtype=dtype))

        self.w_ = nn.Parameter(torch.zeros(N, dtype=dtype))

    def forward(self, inputs):
        '''
        w is the log probability of each input value
        cste should be 0
        '''
        cste = torch.logsumexp(self.w, dim=0)
        return self.w[inputs.squeeze(1)] - cste

class Conditional(NNModule):
    def __init__(self, N, dtype=None):
        super(Conditional, self).__init__(N)
        self.w = nn.Parameter(torch.zeros((N, N), dtype=dtype))
        self.w_ = nn.Parameter(torch.zeros((N, N), dtype=dtype))
    
    def forward(self, inputs, conds):
        conds_ = conds.squeeze(1)
        cste = torch.logsumexp(self.w[conds_], dim=1)
        return self.w[conds_, inputs.squeeze(1)] - cste

class ModelNonCausal(NNModule):
    '''
    The non-causal model captures P(A,B)
    '''
    def __init__(self, N, dtype=None):
        super(ModelNonCausal, self).__init__(N)
        self.w = nn.Parameter(torch.zeros((N, N), dtype=dtype))

    def forward(self, inputs):
        # We need to normalize the weight so that it adds up to 1
        cste = torch.logsumexp(torch.logsumexp(self.w, dim=0), dim=0)
        return self.w[inputs[:,0], inputs[:,1]] - cste

    def initialize_weights(self):
        # Initializing the model parameter to some random value
        pi_A_AND_B = np.random.dirichlet(np.ones(self.N), size=self.N)
        pi_A_AND_B = torch.from_numpy(pi_A_AND_B)        
        self.w.data = torch.log(pi_A_AND_B)

class ModelCausal(NNModule):
    '''
    The causal model captures P(B|A)P(A)
    '''
    def __init__(self, N, dtype=None):
        super(ModelCausal, self).__init__(N)
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

    def initialize_weights(self):
        # Initializing the model parameter to some random value
        pi_A = np.random.dirichlet(np.ones(self.N))
        pi_B_A = np.random.dirichlet(np.ones(self.N), size=self.N)
        pi_A = torch.from_numpy(pi_A)
        pi_B_A = torch.from_numpy(pi_B_A)
        
        self.p_A.w.data = torch.log(pi_A)
        self.p_B_A.w.data = torch.log(pi_B_A)

    def save_weight_snapshot(self):
        self.p_A.save_weight_snapshot()
        self.p_B_A.save_weight_snapshot()

    def load_weight_snapshot(self):
        self.p_A.load_weight_snapshot()
        self.p_B_A.load_weight_snapshot()

class StructuralModel(NNModule):
    def __init__(self, N, dtype=None):
        super(StructuralModel, self).__init__(N)
        self.model_causal = ModelCausal(N, dtype=dtype)
        self.model_noncausal = ModelNonCausal(N, dtype=dtype)
        self.w = nn.Parameter(torch.tensor(0., dtype=dtype))
        self.w_ = nn.Parameter(torch.tensor(0., dtype=dtype))
    
    def initialize_weights(self):
        self.model_causal.initialize_weights()
        self.model_noncausal.initialize_weights()
    
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

    def save_weight_snapshot(self):
        super(StructuralModel, self).save_weight_snapshot()
        self.model_causal.save_weight_snapshot()
        self.model_noncausal.save_weight_snapshot()

    def load_weight_snapshot(self):
        super(StructuralModel, self).save_weight_snapshot()
        self.model_causal.load_weight_snapshot()
        self.model_noncausal.load_weight_snapshot()
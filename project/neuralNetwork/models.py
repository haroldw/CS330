import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class NNModel(nn.Module):
    def __init__(self, N, K, innerDim=32, isConditional=False, isJoint=False):
        # N is the number of unique distinct values each variable can take
        # K is the the number samples per batch
        super(NNModel, self).__init__()
        self.N = N
        self.isConditional = isConditional
        self.isJoint = isJoint
        self.fc1 = nn.Linear(K, innerDim)
        if isConditional:
            self.fc1 = nn.Linear(2*K+N, innerDim)
        elif isJoint:
            self.fc1 = nn.Linear(2*K, innerDim)
        self.fc2 = nn.Linear(innerDim, innerDim)
        outputSize = N
        if isConditional or isJoint:
            outputSize = N * N
        self.fc3 = nn.Linear(innerDim, innerDim)
        self.fc4 = nn.Linear(innerDim, innerDim)
        self.fc5 = nn.Linear(innerDim, innerDim)
        self.fc6 = nn.Linear(innerDim, innerDim)
        self.fc7 = nn.Linear(innerDim, outputSize)
        self.output = torch.nn.Softmax()

    def forward(self, inputs):
        # inputs should have shape (batchSize, K)
        # returns the l2 distance between the ground truth and input data
        x = self.fc1(inputs)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        # x = self.fc4(x)
        # x = F.relu(x)
        # x = self.fc5(x)
        # x = F.relu(x)
        # x = self.fc6(x)
        # x = F.relu(x)
        x = self.fc7(x)
        
        if self.isConditional:
            x = x.reshape((-1, self.N))
            x = F.softmax(x, dim=1)
        elif self.isJoint:
            x = self.output(x)
            x = x.reshape((-1, self.N))
        else:
            x = self.output(x)
        return x

    def save_weight_snapshot(self):
        self.w_ = self.w

    def load_weight_snapshot(self):
        self.w = self.w_

class ModelNonCausal(NNModule):
    '''
    The non-causal model captures P(A,B)
    '''
    def __init__(self, N, K, dtype=None):
        super(ModelNonCausal, self).__init__(N)
        self.p_AB = NNModel(N, K, isJoint=True)

    def forward(self, inputs):
        # We need to normalize the weight so that it adds up to 1
        return self.p_AB(inputs.reshape((1,-1)).float())

    def initialize_weights(self):
        # Initializing the model parameter to some random value
        pi_A_AND_B = np.random.dirichlet(np.ones(self.N), size=self.N)
        pi_A_AND_B = torch.from_numpy(pi_A_AND_B)        
        self.w.data = torch.log(pi_A_AND_B)

class ModelCausal(NNModule):
    '''
    The causal model captures P(B|A)P(A)
    '''
    def __init__(self, N, K, dtype=None):
        super(ModelCausal, self).__init__(N)
        self.p_A = NNModel(N, K)
        self.p_B_A = NNModel(N, K, isConditional=True)

    def forward(self, inputs):
        inputs_A, _ = torch.split(inputs, 1, dim=-1)
        predConditional = self.p_A(inputs_A.reshape((1,-1)).float())
        predMarginal = self.p_B_A(torch.cat((inputs.reshape((1,-1)).float(),
                                             predConditional), dim=1))
        predJoint = torch.mul(predMarginal, predConditional)
        return predConditional, predMarginal, predJoint

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
    def __init__(self, N, K, dtype=None):
        super(StructuralModel, self).__init__(N)
        self.model_causal = ModelCausal(N, K, dtype=dtype)
        self.model_noncausal = ModelNonCausal(N, K, dtype=dtype)
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
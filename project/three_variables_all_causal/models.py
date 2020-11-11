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


class Marginal(NNModule):
    def __init__(self, N, dtype=None):
        super(Marginal, self).__init__(N)
        self.w = nn.Parameter(torch.zeros(N, dtype=dtype))

        self.w_ = nn.Parameter(torch.zeros(N, dtype=dtype))

    def forward(self, inputs):
        """
        w is the log probability of each input value
        cste should be 0
        """
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


class ModelCausal(NNModule):
    """
    The causal model captures P(B|A)P(A)
    """

    def __init__(self, N, dtype=None):
        super(ModelCausal, self).__init__(N)
        self.p_M = Marginal(N, dtype=dtype)
        self.p_C1 = Conditional(N, dtype=dtype)
        self.p_C2 = Conditional(N, dtype=dtype)

    def forward(self, inputs_M, inputs_C1_A, inputs_C1_B, inputs_C2_A, inputs_C2_B):
        return (
            self.p_M(inputs_M)
            + self.p_C1(inputs_C1_B, inputs_C1_A)
            + self.p_C2(inputs_C2_B, inputs_C2_A)
        )

    def save_weight_snapshot(self):
        self.p_M.save_weight_snapshot()
        self.p_C1.save_weight_snapshot()
        self.p_C2.save_weight_snapshot()

    def load_weight_snapshot(self):
        self.p_M.load_weight_snapshot()
        self.p_C1.load_weight_snapshot()
        self.p_C2.load_weight_snapshot()


class StructuralModel(NNModule):
    def __init__(self, N, dtype=None):
        super(StructuralModel, self).__init__(N)
        self.models = {
            # a causes b and a causes c
            "a_ab_ac": ModelCausal(N, dtype=dtype),
            # a causes b and b causes c
            "a_ab_bc": ModelCausal(N, dtype=dtype),
            # a causes c and c causes b
            "a_ac_cb": ModelCausal(N, dtype=dtype),
            "b_ba_bc": ModelCausal(N, dtype=dtype),
            "b_ba_ac": ModelCausal(N, dtype=dtype),
            "b_bc_ca": ModelCausal(N, dtype=dtype),
            "c_ca_cb": ModelCausal(N, dtype=dtype),
            "c_ca_ab": ModelCausal(N, dtype=dtype),
            "c_cb_ba": ModelCausal(N, dtype=dtype)
        }
        model_names = list(self.models.keys())
        model_names.sort()
        self.model_names = model_names
        self.n_models = len(self.models.keys())
        self.w = nn.Parameter(torch.zeros(self.n_models, dtype=dtype))

    def forward(self, inputs):
        logl_models = compute_losses(inputs)
        return self.online_loglikelihood(logl_models)

    def compute_losses(self, inputs):
        inputs_A, inputs_B, inputs_C = torch.split(inputs, 1, dim=1)
        input_dict = {
            "a": inputs_A,
            "b": inputs_B,
            "c": inputs_C
        }
        logl_models = []
        for name in self.model_names:
            m, c1, c2 = name.split("_") # ['a', 'ab', 'ac']
            model = self.models[name]
            log1_model = model(
                inputs_M = input_dict[m],
                inputs_C1_A = input_dict[c1[0]],
                inputs_C1_B = input_dict[c1[1]],
                inputs_C2_A = input_dict[c2[0]],
                inputs_C2_B = input_dict[c2[1]]
            )
            logl_models.append(log1_model)
        return logl_models

    def online_loglikelihood(self, logl_models):
        assert len(logl_models) == self.n_models
        weighted_total = sum(
            [F.logsigmoid(self.w[i]) + torch.sum(logl_models[i]) for i in range(self.n_models)]
        )
        return logsumexp(weighted_total)

    def modules_parameters(self):
        parameters = [self.models[m].parameters() for m in self.model_names]
        return chain.from_iterable(parameters)
    
    def structural_parameters(self):
        return self.w

    def save_weight_snapshot(self):
        super(StructuralModel, self).save_weight_snapshot()
        for model in self.models.values():
            model.save_weight_snapshot()
        
    def load_weight_snapshot(self):
        super(StructuralModel, self).load_weight_snapshot()
        for model in self.models.values():
            model.load_weight_snapshot()

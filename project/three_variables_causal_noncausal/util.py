import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def generate_data_categorical(num_samples, pi_A, pi_B_A, pi_C_B):
    """Sample data using ancestral sampling

    x_A ~ Categorical(pi_A)
    x_B ~ Categorical(pi_B_A[x_A])
    x_C ~ Categorical(pi_C_B[x_B])

    output_shape: (N, 3)
    """
    N = pi_A.shape[0]
    r = np.arange(N)

    x_A = np.dot(np.random.multinomial(1, pi_A, size=num_samples), r)
    x_Bs = np.zeros((num_samples, N), dtype=np.int64)
    x_Cs = np.zeros((num_samples, N), dtype=np.int64)
    for i in range(num_samples):
        x_Bs[i] = np.random.multinomial(1, pi_B_A[x_A[i]], size=1)
    x_B = np.dot(x_Bs, r)
    for i in range(num_samples):
        x_Bs[i] = np.random.multinomial(1, pi_C_B[x_B[i]], size=1)
    x_C = np.dot(x_Cs, r)

    return torch.from_numpy(np.vstack((x_A, x_B, x_C)).T.astype(np.int64))


def logsumexp(a, b):
    min_, max_ = torch.min(a, b), torch.max(a, b)
    return max_ + F.softplus(min_ - max_)


def plot_loss(losses, image_output_path=None):
    num_episodes = losses.shape[-1]
    flat_losses = -losses.reshape((2, -1, num_episodes))
    losses_25, losses_50, losses_75 = np.percentile(flat_losses, (25, 50, 75), axis=1)

    plt.figure(figsize=(9, 5))

    ax = plt.subplot(1, 1, 1)
    ax.plot(losses_50[0], color="C0", label=r"Causal", lw=2)
    ax.fill_between(
        np.arange(num_episodes), losses_25[0], losses_75[0], color="C0", alpha=0.2
    )
    ax.plot(losses_50[1], color="C3", label=r"Non Causal", lw=2)
    ax.fill_between(
        np.arange(num_episodes), losses_25[1], losses_75[1], color="C3", alpha=0.2
    )
    ax.set_xlim([0, 50])
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.legend(loc=4, prop={"size": 13})
    ax.set_xlabel("Number of examples", fontsize=14)
    ax.set_ylabel(r"$\log P(D\mid \cdot \rightarrow \cdot)$", fontsize=14)

    if image_output_path:
        plt.savefig(image_output_path)
    else:
        plt.show()

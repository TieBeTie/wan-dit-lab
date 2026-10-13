import torch


def echo(X):
    # Hover over X: its shape is inferred from the call below.
    return X


sample = torch.zeros(2, 3, 4)
result = echo(sample)

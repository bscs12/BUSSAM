import torch.nn.functional as F


softmax_helper = lambda x: F.softmax(x, 1)

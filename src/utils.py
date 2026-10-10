import random
import numpy as np
import torch

def set_seed(seed=42):
    # fix every random number generator so results are repeatable
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
import torch
from qbdm.qbdm import measure_complexity

x = torch.FloatTensor(16, 16).uniform_(-1, 1)

bin_kc, qbit_kc, _ = measure_complexity(x, bit_depths=[2,4,8])
print(bin_kc, qbit_kc)

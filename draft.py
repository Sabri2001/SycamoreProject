! kubectl exec -ti pytorch-pod -- python

import torch
print(torch.cuda.is_available())

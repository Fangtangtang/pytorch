import torch
import time
device = "cuda"

for i in range(2):
    A = torch.randn(128, 256, device=device, dtype=torch.bfloat16)
    B = torch.randn(256, 512, device=device, dtype=torch.bfloat16)
    C = torch.matmul(A, B)

    torch.cuda.synchronize()
    print("Output shape:", C.shape, C[1,2])
time.sleep(5)

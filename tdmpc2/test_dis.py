from exist_check import _mean_pairwise_distance
import torch
import time

X = torch.randn(256, 512).cuda()

start_time = time.time()
print(_mean_pairwise_distance(X))
end_time = time.time()
print(f"Time taken: {end_time - start_time:.3f} seconds") # 0.139 seconds
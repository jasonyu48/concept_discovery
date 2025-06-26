import numpy as np
import torch

from reptrix import rankme


def test_get_rankme() -> None:
    '''
    Test the RankMe metric, which is the measure of the quality of the embeddings. It takes a batch of emcoding vectors. Low value indicates representation collapse.
    '''
    np.random.seed(0)
    torch.manual_seed(0)
    activations_arr = torch.randn(30000, 512, device="cuda")

    metric_rankme = rankme.get_rankme(activations_arr)
    print(f"RankMe: {metric_rankme}")
    error = (512-metric_rankme)/512
    if error < 0.01:
        print("Success")
    else:
        print("Failure")
    print(f"Error: {error}")



if __name__ == "__main__":
    test_get_rankme()
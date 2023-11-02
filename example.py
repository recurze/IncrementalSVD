from isvd import isvd_thin
from sklearn.utils.extmath import randomized_svd
import numpy as np
import scipy as sp
import time


def generate_low_rank_matrix(m, n, k):
    return np.random.normal(size=(m, k)) @ np.random.normal(size=(k, n))


def generate_noise(m, n, density):
    return sp.sparse.random(m, n, density=density, random_state=0).toarray()


def generate_report(svd, M, **kwargs):
    start = time.time()
    U, s, Vh = svd(M, **kwargs)
    print(f"{svd.__name__}{(kwargs)} Time: {time.time() - start} \n"
          f"Product Error     : {np.linalg.norm(M - (U @ np.diag(s) @ Vh))}\n"
          f"Orthogonal Error U: {np.linalg.norm(np.eye(U.shape[1]) - (U.T @ U))}\n"
          f"Orthogonal Error V: {np.linalg.norm(np.eye(Vh.shape[0]) - (Vh @ Vh.T))}\n")


def run(M, ranks=None):
    if ranks is None:
        ranks = [M.shape[1]]

    generate_report(np.linalg.svd, M, full_matrices=False)
    for rank in ranks:
        generate_report(isvd_thin, M, n_components=rank)

    for rank in ranks:
        generate_report(randomized_svd, M, n_components=rank)


def test_pure_low_rank():
    m, n, k = 1000000, 100, 15
    M = generate_low_rank_matrix(m, n, k)
    generate_report(np.linalg.svd, M, full_matrices=False)
    generate_report(isvd_thin, M)
    generate_report(randomized_svd, M, n_components=20)


def test_noisy_low_rank():
    m, n, k = 1000000, 100, 10
    M = generate_low_rank_matrix(m, n, k) + generate_noise(m, n, 0.05)
    generate_report(np.linalg.svd, M, full_matrices=False)

    for rank in range(10, 100, 20):
        generate_report(isvd_thin, M, n_components=rank)

    for rank in range(10, 100, 20):
        generate_report(randomized_svd, M, n_components=rank)

if __name__ == "__main__":
    test_pure_low_rank()
    #test_noisy_low_rank()

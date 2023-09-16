from functools import reduce
import numpy as np


def svd_K(W, K):
    U, S, V = np.linalg.svd(W, full_matrices=False)

    m, n = W.shape
    m1_num = m * n
    m2_num = K * (m + n + 1)

    W_ = reduce(np.dot, [U[:, :K], np.diag(S[:K]), V[:K]])
    return W_, m1_num, m2_num


def vh_decompose(param_name, W, K):
    vh_cache = {}
    N, C, D, _ = W.shape
    W = W.transpose(1, 2, 3, 0).reshape((C * D, D * N))
    try:
        U, S, V = vh_cache[param_name]
    except KeyError:
        U, S, V = np.linalg.svd(W)
        vh_cache[param_name] = (U, S, V)
    v = U[:, :K] * np.sqrt(S[:K])
    v = v.reshape((C, D, 1, K)).transpose(3, 0, 1, 2)
    h = V[:K, :] * np.sqrt(S)[:K, None]
    h = h.reshape((K, 1, D, N)).transpose(3, 0, 1, 2)

    m1_num = np.prod(W.shape)
    m2_num = np.prod(v.shape) + np.prod(h.shape)

    return v, h, m1_num, m2_num, m2_num / m1_num


def ratio_information(s, ratio):
    """ Function that computer the number of components to be kept based on
    the provided ratio

    Args:
        s: singular values
        ratio: number that corresponds to the information we want to keep (
        min:0, max:1)

    Return:
        n_components: number of components correspond to the given ratio
    """

    explained_variance_ = s
    total_var = explained_variance_.sum()
    explained_variance_ratio_ = explained_variance_ / total_var
    ratio_cumsum = np.cumsum(explained_variance_ratio_)
    n_components = np.searchsorted(ratio_cumsum, ratio)

    return n_components


def reconstruct_svd(X, ratio):
    """ Reconstruct the rank-K version of the given matrix

    Args:
        X (ndarray): give matrix
        ratio (float, 0< ratio <1): information we want to keep after
        reconstruction.

    Return:
        rank-K version of the input matrix.
    """
    [U, S, V] = np.linalg.svd(X, full_matrices=False)
    svp = ratio_information(S, ratio)
    S_svp = np.diag(S[:svp]).astype(np.float32)

    A, B = X.shape
    m1_num = A * B
    m2_num = svp * (A + B + 1)
    return np.dot(U[:, :svp], np.dot(S_svp, V[:svp])), m1_num, m2_num

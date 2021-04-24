import numpy as np


def texify_augmented_matrix(aug):
    n, m = aug.shape
    s = '\\begin{equation}\n\\begin{bmatrix}'''
    t = '\n\\end{bmatrix}\n\\end{equation}'
    shape = f'[{n * "r"}|{(m - n) * "r"}]\n'
    rows = '\\\\\n'.join(n * [(' & '.join(m * ['{}']))]).format(*aug.flatten())
    return s + shape + rows + t


def gauss_jordan_elimination(aa, b):
    aa, b = gaussian_elimination(aa, b, allow_pivot=True)
    aa, b = gaussian_elimination(aa[::-1, ::-1], b[::-1, :], allow_pivot=False)
    return aa[::-1, ::-1], b[::-1, :]


def gaussian_elimination(aa, b=None, allow_pivot=True):
    aug = aa if b is None else np.hstack([aa, b])
    rank, _ = aa.shape

    def pivot():
        pivot_index = abs(aug[i:, i]).argmax()
        if pivot_index != 0:
            aug[[i, pivot_index + i]] = aug[[pivot_index + i, i]]
            aug[i] *= -1

    for i in range(rank - 1):
        if allow_pivot:
            pivot()
        diff = (aug[i + 1:, i:i + 1] / aug[i, i]) @ aug[i:i + 1, :]
        aug[i + 1:] -= diff

    return aug[:, :rank], aug[:, rank:]


def determinant(aa):
    aa, _ = gaussian_elimination(aa)
    return np.diagonal(aa).prod()


def solve_system(aa, b):
    aa, b = gauss_jordan_elimination(aa, b)
    x = b / aa.diagonal()[:, np.newaxis]
    return x.T


def matrix_inverse(aa):
    rank, _ = aa.shape
    aa, bb = gauss_jordan_elimination(aa, np.eye(rank, dtype=aa.dtype))
    return bb / aa.diagonal()[:, np.newaxis]


def matrix_pseudoinverse(aa):
    return matrix_inverse(aa.T @ aa) @ aa.T

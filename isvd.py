import numpy as np


def isvd_thin(M, tol=1e-9, n_components=500):
    m, n = M.shape

    # init: UsV = M[:, 0]
    s = np.array([np.linalg.norm(M[:, 0])])
    U = M[:, 0].reshape((-1, 1))/s
    Vh = np.eye(1)

    U_dash = np.eye(1)

    for i in range(1, n):
        c = M[:, i]
        utc = U.T.dot(c)
        ul = U.dot(utc)
        l = U_dash.T.dot(utc)
        j = c - ul
        k = np.linalg.norm(j)
        j /= k

        if k < tol:
            k = 0

        Q = np.block([
            [np.diag(s), l.reshape((s.size, 1))],
            [np.zeros((1, s.size)),  np.array([k])]
        ])
        A, s_dash, Bh = np.linalg.svd(Q)

        rank = U.shape[1]
        if k == 0 or rank >= n_components:
            s = s_dash[:-1]

            U_dash = U_dash @ A[:-1, :-1]
            Vh = Bh[:-1, :] @ np.block([
                [Vh, np.zeros((Vh.shape[0], 1))],
                [np.zeros((1, Vh.shape[1])), np.array([1])]
            ])
        else:
            s = s_dash

            U = np.block([U, j.reshape((m, 1))])
            U_dash = np.block([
                [U_dash, np.zeros((U_dash.shape[0], 1))],
                [np.zeros((1, U_dash.shape[1])), np.array([1])]
            ]) @ A

            Vh = Bh @ np.block([
                [Vh, np.zeros((rank, 1))],
                [np.zeros((1, i)), np.array([1])]
            ])

    return U @ U_dash, s, Vh

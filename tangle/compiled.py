import numba as nb
import numpy as np


@nb.njit(cache=True)
def n_bitstrings_two_or_fewer_consecutive_1s(n):
    """Count the n-bit strings containing two or fewer consecutive 1 bits"""
    # ap = np.cbrt(19 + 3 * np.sqrt(33))
    # am = np.cbrt(19 - 3 * np.sqrt(33))
    ap = 3.3090564799660953
    am = 1.2088037856763885

    bp = (1 + 1j * np.sqrt(3)) / 2
    bm = (1 - 1j * np.sqrt(3)) / 2

    ll = [
        (1 + ap + am) / 3,
        (1 - bp * am - bm * ap) / 3,
        (1 - bm * am - bp * ap) / 3,
    ]

    def _w(x):
        return np.array([x, x, x * (x - 1), 1])

    def _a(x, y, z):
        return (y * z - y - z + 2) / ((x - y) * (x - z))

    def _v(p):
        return _w(p[0]) * _a(p[0], p[1], p[2]) * p[0] ** (n - 2)

    ans = np.sum(_v(ll) + _v(np.roll(ll, 1)) + _v(np.roll(ll, 2))).real

    return int(ans + 0.5)


@nb.njit(cache=True)
def find_rem(c):
    length = c * 2
    p = np.empty(length, dtype=np.int_)

    rem = np.empty((n_bitstrings_two_or_fewer_consecutive_1s(length), 2), dtype=np.uint64)
    rem_index = 0

    idx = 0
    while idx < 2 ** length:
        more_than_two_consecutive_1s = idx & (idx << 1) & (idx << 2)
        if more_than_two_consecutive_1s:
            idx += more_than_two_consecutive_1s
            b = 0
            while more_than_two_consecutive_1s:
                more_than_two_consecutive_1s >>= 1
                b += 1
            idx = (idx >> b) << b
        else:
            x, y, u, v, w = 1, 0, 0, 0, 1
            for k in range(length):
                p[k] = x + length * y
                if idx & (1 << k):
                    x, y = u + w * (v - y), v + w * (x - u)
                else:
                    x, y, u, v, w = 2 * x - u + w * (v - y), 2 * y - v + w * (x - u), 2 * x - u, 2 * y - v, -w

            if (x != 1 or y != 0) and np.unique(p).size == length:
                dist = 2 * ((x - 1) ** 2 + y ** 2)
                if w == 1:
                    dist = dist + 1
                if y == v:
                    dist = dist + 2
                rem[rem_index] = idx, dist
                rem_index += 1

            idx += 1

    return rem[:rem_index]


@nb.njit(cache=True)
def pm_to_lr(pm, n):
    lr = np.uint64(1)
    for _ in range(n - 1):
        last_bit = lr ^ ~(pm & np.uint64(1))
        lr = (lr << np.uint64(1)) | (last_bit & np.uint64(1))
        pm >>= np.uint64(1)
    return lr


@nb.njit(cache=True)
def popcount(x):
    """Count the nonzero bits in x

    https://en.wikipedia.org/wiki/Hamming_weight#Efficient_implementation"""
    m1 = np.uint64(0x5555555555555555)
    m2 = np.uint64(0x3333333333333333)
    m4 = np.uint64(0x0f0f0f0f0f0f0f0f)
    h01 = np.uint64(0x0101010101010101)
    x -= (x >> np.uint64(1)) & m1
    x = (x & m2) + ((x >> np.uint64(2)) & m2)
    x = (x + (x >> np.uint64(4))) & m4
    return (x * h01) >> np.uint64(56)


@nb.njit(cache=True)
def one_turn(n, idx):
    lr = pm_to_lr(idx, n)
    return np.abs(n - 2 * popcount(lr)) == 4


@nb.njit(cache=True)
def compute_tangle_points(n, idx, p, xyuvw, k0, k1):
    x, y, u, v, w = xyuvw
    for k in range(k0, k1):
        p[k] = x + n * y
        a = w * (v - y)
        b = w * (x - u)
        if not idx & np.uint64(1):
            u = 2 * x - u
            v = 2 * y - v
            w = -w
        x = a + u
        y = b + v
        idx >>= np.uint64(1)
    return x, y, u, v, w


@nb.njit(cache=True)
def count_tangles(n, rem_list, prefix_list):
    count = 0
    n_half = n // 2
    m = np.uint64(2 ** n - 1)
    idx_roll = np.empty(2 * n, dtype=np.uint64)
    p = np.empty(n, dtype=np.int_)
    idx_min = np.uint64(2 ** (n - 2))

    post = np.empty_like(rem_list[:, 0])

    # By definition, no prefix has dist == 0
    # Therefore, the first if statement in the loop
    # will always be executed on the first iteration
    dist = np.uint64(0)

    for prefix, prefix_dist in prefix_list:
        # Select postfixes with distances to match the current prefix
        if dist != prefix_dist:
            dist = prefix_dist
            post = rem_list[rem_list[:, 1] == dist][:, 0]

        xyuvw0 = (1, 0, 0, 0, 1)
        xyuvw = compute_tangle_points(n, prefix, p, xyuvw0, 0, n_half)

        # Shift the prefix to the left and insert the postfix
        pref = prefix << np.uint64(n_half)
        idx_list = pref + post

        for idx in idx_list:
            if one_turn(n, idx):
                xyuvw_final = compute_tangle_points(n, idx, p, xyuvw, n_half, n)
                if xyuvw_final == xyuvw0 and np.unique(p).size == n:
                    xdi = np.uint64(0)
                    for j in np.arange(n, dtype=np.uint64):
                        xdi = (xdi << np.uint64(1)) | ((idx >> j) & np.uint64(1))
                    for j in np.arange(n, dtype=np.uint64):
                        nmj = n - j
                        idx_roll[j] = (idx >> j) | ((idx << nmj) & m)
                        idx_roll[n + j] = (xdi >> j) | ((xdi << nmj) & m)
                    if idx == np.min(idx_roll[idx_roll > idx_min]):
                        count += 1
    return count

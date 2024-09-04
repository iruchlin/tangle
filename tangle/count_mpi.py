from mpi4py import MPI
import numpy as np
import sys
import time

from compiled import find_rem, count_tangles
from truth import n_tangles


# Initialize the MPI interface
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
root = 0

# Get the tangle class from the first command line argument
c = int(sys.argv[1])
assert 1 <= c <= 16

# The length of the tangle
n = 4 * c

if rank == root:
    print(f"class {c}")
    print(f"length {n}")

# Start measuring the time to precompute
start_time = time.time()

if rank == root:
    # Precompute the tangle prefixes in the range [idx_min_half, idx_max_half]
    # The prefixes are half of the tangle length, and thus have n/2 bits
    # - The lower bound has the pattern 0100_0000_0000_0000,
    #   where the second bit is 1 and all others are 0
    # - The upper bound has the pattern 0110_1101_1011_0110,
    #   where the pattern 011 repeats to fill the bits
    idx_min_half = 2 ** (n // 2 - 2)
    idx_max_half = int("0b" + ("011" * ((n // 3) + 1))[:n//2], 2)

    rem_list_root = find_rem(c)
    len_rem_list = len(rem_list_root)
    prefix_list_root = rem_list_root[
        (rem_list_root[:, 0] >= idx_min_half) &
        (rem_list_root[:, 0] <= idx_max_half)
    ]

    rng = np.random.default_rng(1234)
    rng.shuffle(prefix_list_root)

    len_prefix_list = len(prefix_list_root)
else:
    rem_list_root = None
    len_rem_list = 0
    prefix_list_root = None
    len_prefix_list = 0

len_rem_list = comm.bcast(len_rem_list, root=root)
len_prefix_list = comm.bcast(len_prefix_list, root=root)

win_rem_list = MPI.Win.Allocate_shared(
    2 * len_rem_list * MPI.UINT64_T.Get_size(),
    MPI.UINT64_T.Get_size(),
    comm=comm,
)
rem_buf, _ = win_rem_list.Shared_query(root)
rem_list = np.ndarray(
    buffer=rem_buf,
    dtype=np.uint64,
    shape=(len_rem_list, 2),
)

win_prefix_list = MPI.Win.Allocate_shared(
    2 * len_prefix_list * MPI.UINT64_T.Get_size(),
    MPI.UINT64_T.Get_size(),
    comm=comm,
)
prefix_buf, _ = win_prefix_list.Shared_query(root)
prefix_list = np.ndarray(
    buffer=prefix_buf,
    dtype=np.uint64,
    shape=(len_prefix_list, 2),
)

if rank == root:
    np.copyto(rem_list, rem_list_root)
    rem_list_root = None
    np.copyto(prefix_list, prefix_list_root)
    prefix_list_root = None

index_list = np.zeros((size, 2), dtype=np.int_)
index_list[0, 1] = len_prefix_list // size + 1
for r in range(1, size):
    index_list[r, 0] = index_list[r - 1, 1]
    offset = 1 if r < len_prefix_list % size else 0
    index_list[r, 1] = index_list[r, 0] + len_prefix_list // size + offset

index_lower = index_list[rank, 0]
index_upper = index_list[rank, 1]

comm.Barrier()

prefix_list[index_lower:index_upper] = prefix_list[index_lower:index_upper][
        np.argsort(prefix_list[index_lower:index_upper, 1], kind="stable")
]

comm.Barrier()

if rank == root:
    print(f"precompute time {time.time() - start_time}")

start_time = time.time()

t_count = count_tangles(n, rem_list, prefix_list[index_lower:index_upper])
print(
    f"rank {rank}, "
    f"index [{index_lower}, {index_upper}], "
    f"tangles {t_count}, "
    f"time {time.time() - start_time}"
)
total_t_count = comm.reduce(t_count, op=MPI.SUM, root=root)

if rank == root:
    print(f"n_tangles {total_t_count}")
    print(f"Correct: {total_t_count == n_tangles[c]}")
    print(f"time {time.time() - start_time}")

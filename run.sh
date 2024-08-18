
source .venv/bin/activate
mpirun -n 2 python3 tangle/count_mpi.py $1

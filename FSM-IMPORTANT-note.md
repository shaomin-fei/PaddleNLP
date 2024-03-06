1. Get  error when inferencing (Linux):
Intel MKL function load error: cpu specific dynamic library is not loaded
FIx it by two steps:
a. export OMP_NUM_THREADS=1 (maybe not necessary)
b. conda install nomkl
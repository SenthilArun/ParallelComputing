nvcc -o MultiGpu_NCCL_MPI.cu \
     -I/usr/include \
     -L/usr/lib/x86_64-linux-gnu \
     -lnccl -lcublas -lcurand \
     -std=c++14 -O3 -arch=sm_86

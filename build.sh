# cuBLAS benchmarks:
rm -rf build
mkdir build

nvcc -arch=sm_80 -O3 -std=c++17 -Ibench \
     bench/kernels/gemm-mma.cu bench/check-gemm.cu -lcublas -o build/check-gemm

# running benchmarks:
./build/check-gemm; echo "exit: $?"
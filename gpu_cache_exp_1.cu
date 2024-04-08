#include <iostream>
#include <cuda_runtime.h>

static const int _warpSize = 32;

__global__ void cacheKernel(int* array_ptr) {
    int* thdPtr = array_ptr + threadIdx.x;
    int x = __ldca(thdPtr); // L1 Cache Miss, Load a cache line into L1
    __stwb(thdPtr, x + threadIdx.x);
    int complement_tid = _warpSize - 1 - threadIdx.x;
    int* complementPtr = array_ptr + complement_tid;
    int _x = __ldca(complementPtr);
    __stwb(thdPtr, _x + threadIdx.x);
}

void printHost(int* array) {
    using namespace std;
    cout << "Array: [ ";
    for (int i = 0; i < _warpSize; i++) {
        cout << array[i] << ' ';
    }
    cout << "]" << std::endl;
}

int main(void) {
    int* _host;
    int* _dev;
    _host = (int*)malloc(_warpSize * sizeof(int));
    memset(_host, 0, _warpSize * sizeof(int));
    cudaMalloc((void**)&_dev, _warpSize * sizeof(int));
    cudaMemcpy(_dev, _host, _warpSize * sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid(1, 1, 1);
    dim3 block(_warpSize, 1, 1);
    cacheKernel<<<grid, block>>>(_dev);
    cudaDeviceSynchronize();

    cudaMemcpy(_host, _dev, _warpSize * sizeof(int), cudaMemcpyDeviceToHost);
    printHost(_host);
    cudaFree(_dev);
    free(_host);
}

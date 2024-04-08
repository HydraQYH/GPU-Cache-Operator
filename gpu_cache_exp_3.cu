#include <iostream>
#include <unistd.h>
#include <cuda_runtime.h>

static const int _warpSize = 32;
static const int _pageSize = 4096;

__device__ uint get_smid(void) {
    uint ret;
    asm("mov.u32 %0, %smid;" : "=r"(ret));
    return ret;
}

__global__ void cacheKernel(int* array_ptr, int* locked, int* result_ptr) {
    int* thdPtr = array_ptr + threadIdx.x;
    if (blockIdx.x == 0) {
        int x = __ldca(thdPtr); // Load cache line to SM0's L1 Cache

        if (threadIdx.x == 0) {
            // *locked == 0: set *locked to 2 and return 0
            // *locked != 0: return *locked and condition True, execute while loop
            while (atomicCAS(locked, 0, 2) != 0);
        }
        __syncthreads();

        __stwb(thdPtr, x + threadIdx.x);    // Flush to L2 Cache?
        atomicExch(locked, 1);  // set *locked to 1
    } else if (blockIdx.x == 1) {
        if (threadIdx.x == 0) {
            // *locked == 1: set *locked to 2 and return 1
            // *locked != 1: return *locked and condition True, execute while loop
            while (atomicCAS(locked, 1, 2) != 1);
        }
        __syncthreads();

        int _x = __ldcg(thdPtr);    // Load from L2

        result_ptr[threadIdx.x + _warpSize] = _x;
    }

#ifdef LOG_BLOCK_ON_SM
    if (threadIdx.x == 0) {
        printf("Block %d on SM %u\n", blockIdx.x, get_smid());
    }
#endif
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
    _host = (int*)malloc(_warpSize * sizeof(int));
    memset(_host, 0, _warpSize * sizeof(int));

    int* _dev;
    cudaMalloc((void**)&_dev, _pageSize);
    cudaMemcpy(_dev, _host, _warpSize * sizeof(int), cudaMemcpyHostToDevice);

    int* _lock;
    cudaMalloc((void**)&_lock, _pageSize);
    cudaMemset(_lock, 0, _pageSize);

    int* _res;
    cudaMalloc((void**)&_res, _pageSize);
    cudaMemset(_res, 0, _pageSize);

    dim3 grid(2, 1, 1);
    dim3 block(_warpSize, 1, 1);
    cacheKernel<<<grid, block>>>(_dev, _lock, _res);
    cudaDeviceSynchronize();
    
    cudaMemcpy(_host, _dev, _warpSize * sizeof(int), cudaMemcpyDeviceToHost);
    printf("TB0 Write -- ");
    printHost(_host);

    // cudaMemcpy(_host, _res, _warpSize * sizeof(int), cudaMemcpyDeviceToHost);
    // printf("TB1 1st Read -- ");
    // printHost(_host);

    cudaMemcpy(_host, _res + _warpSize, _warpSize * sizeof(int), cudaMemcpyDeviceToHost);
    printf("TB1 Read -- ");
    printHost(_host);

    cudaFree(_dev);
    cudaFree(_lock);
    cudaFree(_res);
    free(_host);
}


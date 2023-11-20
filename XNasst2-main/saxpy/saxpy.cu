#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

extern float toBW(int bytes, float sec);

__global__ void
saxpy_kernel(int N, float alpha, float* x, float* y, float* result) {

    // compute overall index from position of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N)
       result[index] = alpha * x[index] + y[index];
}

void
saxpyCuda(int N, float alpha, float* xarray, float* yarray, float* resultarray) {

    int totalBytes = sizeof(float) * 3 * N;

    // compute number of blocks and threads per block
    const int threadsPerBlock = 512;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* device_x;
    float* device_y;
    float* device_result;
    // TODO allocate device memory buffers on the GPU using cudaMalloc分配设备内存缓冲器
    cudaMalloc((void**)&device_x, sizeof(float) * N);
    cudaMalloc((void**)&device_y, sizeof(float) * N);
    cudaMalloc((void**)&device_result, sizeof(float) * N);
    
    // start timing after allocation of device memory分配设备后开始计时
    double startTime1 = CycleTimer::currentSeconds();
    
    // TODO copy input arrays to the GPU using cudaMemcpy将输入数组复制到Gpu
    cudaMemcpy(device_x, xarray, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, yarray, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start); // 记录开始时间
     // run kernel运行
    saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y, device_result);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); // 记录结束时间
    cudaEventSynchronize(stop); // 等待所有操作完成
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop); // 计算时间差
    // TODO copy result from GPU using cudaMemcpy复制结果从Gpu
     cudaMemcpy(resultarray, device_result, sizeof(float) * N, cudaMemcpyDeviceToHost);
    // end timing after result has been copied back into host memory结果复制会主机内存后结束计时
    double endTime1 = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    double overallDuration1 = endTime1 - startTime1;
    printf("Overall: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration1, toBW(totalBytes, overallDuration1));
   double overallDuration = milliseconds; // 核函数执行时间
    printf("Kernel Execution Time: %.3f ms\n", overallDuration);
    // TODO free memory buffers on the GPU释放内存缓冲区
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_result);
    
}

void
printCudaInfo() {

    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}

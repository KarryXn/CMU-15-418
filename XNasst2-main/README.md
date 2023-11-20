# CMU 15-418/618, Fall 2023

# Assignment 2

This is the starter code for Assignment 2 of CMU class 15-418/618, Fall 2023

Please review the course's policy on [academic
integrity](http://www.cs.cmu.edu/~418/academicintegrity.html),
regarding your obligation to keep your own solutions private from now
until eternity.



**CUDA热身1:SAXPY(5分)**

为了获得一些编写CUDA程序的练习，您的热身任务是实现SAXPY函数。这是一个在矩阵数学库中常见的函数。对于输入数组x和y，输出数组dest和值a(均为单精度浮点值)，该函数计算dest[i] = a*x[i] +

2

(名称“SAXPY”表示“单精度a乘以x加y”)这部分赋值的起始代码位于SAXPY目录中。

在SAXPY .cu中的saxpyCuda()函数中完成SAXPY的实现。您将需要分配设备全局内存数组并插入调用以在主机和设备内存之间移动数据。这些问题将在程序员指南的3.2.2节中讨论。

作为实现的一部分，在saxpyCuda()中为CUDA内核调用添加计时器。添加之后，你的程序应该为两次执行计时:

•提供的启动器代码包含计时器，用于测量将数据复制到GPU，运行内核和将数据复制回CPU的**整个过程**。

•计时器应该只测量运行内核所花费的时间。(它们不应该包括CPU到GPU数据传输或将结果传输回CPU的时间。)

**在添加计时代码时，请注意**:CUDA内核在GPU上的执行与主应用程序线程在CPU上的运行是异步的。你应该在内核调用之后调用cudaThreadSynchronize()来等待GPU上所有CUDA工作的完成。此调用仅在GPU上的所有先前CUDA工作完成时返回。(不等待GPU完成，你的CPU计时器将报告基本上没有时间流逝!)请注意，在包括将数据传输回CPU的时间的测量中，在最终计时器之前(在调用cudaMemcpy()将数据返回给CPU之后)*不*需要调用cudaThreadSynchronize()，因为cudaMemcpy()在复制完成之前不会返回到调用线程。

**cudaThreadSynchronize()函数已被弃用**；

## Found 3 CUDA devices
Device 0: Tesla T4
SMs:        40
Global mem（总体存储器）: 14910 MB
CUDA Cap:   7.5
Device 1: Tesla T4
SMs:        40
Global mem: 14910 MB
CUDA Cap:   7.5
Device 2: Tesla T4
SMs:        40
Global mem: 14910 MB
CUDA Cap:   7.5

Overall: 50.240 ms              [4.449 GB/s]
Kernel Execution Time: 0.913 ms
Overall: 55.650 ms              [4.016 GB/s]
Kernel Execution Time: 0.914 ms
Overall: 55.600 ms              [4.020 GB/s]

## Found 3 CUDA devices
Device 0: Tesla T4
SMs:        40
Global mem: 14910 MB
CUDA Cap:   7.5
Device 1: Tesla T4
SMs:        40
Global mem: 14910 MB
CUDA Cap:   7.5
Device 2: Tesla T4
SMs:        40
Global mem: 14910 MB
CUDA Cap:   7.5

## Found 3 CUDA devices
Device 0: Tesla T4
SMs:        40
Global mem: 14910 MB
CUDA Cap:   7.5
Device 1: Tesla T4
SMs:        40
Global mem: 14910 MB
CUDA Cap:   7.5
Device 2: Tesla T4
SMs:        40
Global mem: 14910 MB
CUDA Cap:   7.5

Overall: 75.664 ms              [2.954 GB/s]
Kernel Execution Time: 0.921 ms
Overall: 63.300 ms              [3.531 GB/s]
Kernel Execution Time: 0.916 ms
Overall: 55.842 ms              [4.003 GB/s]
Kernel Execution Time: 0.918 ms

---

## Found 3 CUDA devices
Device 0: Tesla T4
SMs:        40
Global mem: 14910 MB
CUDA Cap:   7.5
Device 1: Tesla T4
SMs:        40
Global mem: 14910 MB
CUDA Cap:   7.5
Device 2: Tesla T4
SMs:        40
Global mem: 14910 MB
CUDA Cap:   7.5

## Overall: 75.664 ms              [2.954 GB/s]
Kernel Execution Time: 0.921 ms
Overall: 63.300 ms              [3.531 GB/s]
Kernel Execution Time: 0.916 ms
Overall: 55.842 ms              [4.003 GB/s]
Kernel Execution Time: 0.918 ms
zyo@350659f1375b:~/XNasst2-main/saxpy$ make
mkdir -p objs/
nvcc saxpy.cu -O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc -c -o objs/saxpy.o
g++ -m64 -O3 -Wall -o cudaSaxpy objs/main.o  objs/saxpy.o -L/usr/local/cuda-11.3/lib64/ -lcudart
zyo@350659f1375b:~/XNasst2-main/saxpy$ ./cudaSaxpy

## Found 3 CUDA devices
Device 0: Tesla T4
SMs:        40
Global mem: 14910 MB
CUDA Cap:   7.5
Device 1: Tesla T4
SMs:        40
Global mem: 14910 MB
CUDA Cap:   7.5
Device 2: Tesla T4
SMs:        40
Global mem: 14910 MB
CUDA Cap:   7.5

Overall: 60.972 ms              [3.666 GB/s]
Kernel Execution Time: 0.914 ms
Overall: 55.586 ms              [4.021 GB/s]
Kernel Execution Time: 0.912 ms
Overall: 55.524 ms              [4.026 GB/s]
Kernel Execution Time: 0.911 ms

**问题:**比较和解释两组计时器(你添加的计时器和已经在提供的启动器代码中的计时器)提供的结果之间的差异。观察到的带宽值是否与报告的机器不同组件可用的带宽大致一致?*提示*:您应该使用web来跟踪NVIDIA RTX 2080 GPU的内存带宽，以及计算机PCIe-x16总线的最大传输速度。它是PCIe 3.0, 16通道总线连接CPU和GPU。

差异：未包括CPU到GPU数据传输或将结果传输回CPU的时间

```C
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

```


## CUDA2:平行前缀和

---

初始make

## Found 3 CUDA devices
Device 0: Tesla T4
SMs:        40
Global mem: 14910 MB
CUDA Cap:   7.5
Device 1: Tesla T4
SMs:        40
Global mem: 14910 MB
CUDA Cap:   7.5
Device 2: Tesla T4
SMs:        40
Global mem: 14910 MB
CUDA Cap:   7.5

## Scan Score Table:./checker.pl -m scan

---

## | Element Count   | Target Time     | Your Time       | Score           |

## | 10000           |                 | 0.007           | 0               |
| 100000          |                 | 0.004           | 0               |
| 1000000         |                 | 0.009           | 0               |
| 2000000         |                 | 0.009           | 0               |

## |                                   | Total score:    | 0/5             |

## Scan Score Table:./checker.pl find_peaks

---

## | Element Count   | Target Time     | Your Time       | Score           |

## | 10000           |                 | 0.004           | 0               |
| 100000          |                 | 0.008           | 0               |
| 1000000         |                 | 0.012           | 0               |
| 2000000         |                 | 0.009           | 0               |

## |                                   | Total score:    | 0/5             |




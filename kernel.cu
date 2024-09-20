#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
using namespace std;
#define N (1024 * 1024)

__global__ void kernel(float* data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float x = 2 * 3.1415926 * (float)idx / (float)N;
	data[idx] = sinf(sqrtf(x));
}
int main(int argc, char* argv[])
{
	// Начало измерения времени
	clock_t start_time = clock();

	float* a = (float*)malloc(N * sizeof(float));
	float* dev = nullptr;

	// выделить память на GPU
	cudaMalloc((void**)&dev, N * sizeof(float));

	// Получаем информацию об устройстве
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);  // Используем первый GPU
	int numMultiprocessors = devProp.multiProcessorCount;
	int maxThreadsPerBlock = devProp.maxThreadsPerBlock;

	// Вычисление количества блоков
	int blocksPerMultiprocessor = maxThreadsPerBlock / maxThreadsPerBlock;
	int numBlocks = numMultiprocessors * blocksPerMultiprocessor;

	// Конфигурация запуска
	kernel << <dim3(blocksPerMultiprocessor), dim3(maxThreadsPerBlock) >> > (dev);

	// конфигурация запуска N нитей
	//kernel << <dim3((N / 512), 1), dim3(512, 1) >> > (dev);
	// скопировать результаты в память CPU
	cudaMemcpy(a, dev, N * sizeof(float), cudaMemcpyDeviceToHost);
	// освободить выделенную память
	cudaFree(dev);
	
	//for (int idx = 0; idx < N; idx++)
	//	printf("a[%d] = %.5f\n", idx, a[idx]);
	free(a);
	
	int deviceCount;
	//cudaDeviceProp devProp;
	cudaGetDeviceCount(&deviceCount);
	printf("Found %d devices\n", deviceCount);
	for (int device = 0; device < deviceCount; device++)
	{
		cudaGetDeviceProperties(&devProp, device);
		printf("Device %d\n", device);
		printf("Compute capability : %d.%d\n", devProp.major, devProp.minor);
		printf("Name : %s\n", devProp.name);
		printf("Total Global Memory : %u\n", devProp.totalGlobalMem);
		printf("Shared memory per block: %d\n", devProp.sharedMemPerBlock);
		printf("Registers per block : %d\n", devProp.regsPerBlock);
		printf("Warp size : %d\n", devProp.warpSize);
		printf("Max threads per block : %d\n", devProp.maxThreadsPerBlock);
		printf("Total constant memory : %d\n", devProp.totalConstMem);
		printf("MultiProcessor count : %d\n", devProp.multiProcessorCount);
		printf("Kernel Exec Timeout Enabled : %d\n", numBlocks);
		printf("Kernel Exec Timeout Enabled : %d\n", devProp.kernelExecTimeoutEnabled);
	}

	// Конец измерения времени
	clock_t end_time = clock();

	// Разница во времени
	double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

	printf("Время выполнения: %f секунд\n", elapsed_time);

	return 0;
}
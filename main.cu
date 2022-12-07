#include <stdio.h>
#include <stdint.h>

#include "support.h"
#include "kernel.cu"

int main(int argc, char* argv[])
{
    Timer timer;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    unsigned int *in_h1;
    unsigned int* bins_h1;
    unsigned int *in_d1;
    unsigned int* bins_d1;
    unsigned int *in_d2;
    unsigned int* bins_d2;
    unsigned int num_elements, num_bins;
    cudaError_t cuda_ret;
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    

    if(argc == 1) {
        num_elements = 1000000;
        num_bins = 4096;
    } else if(argc == 2) {
        num_elements = atoi(argv[1]);
        num_bins = 4096;
    } else if(argc == 3) {
        num_elements = atoi(argv[1]);
        num_bins = atoi(argv[2]);
    } else {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./histogram            # Input: 1,000,000, Bins: 4,096"
           "\n    Usage: ./histogram <m>        # Input: m, Bins: 4,096"
           "\n    Usage: ./histogram <m> <n>    # Input: m, Bins: n"
           "\n");
        exit(0);
    }
    initVector(&in_h, num_elements, num_bins);
    bins_h = (unsigned int*) malloc(num_bins*sizeof(unsigned int));

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    Input size = %u\n    Number of bins = %u\n", num_elements,
        num_bins);
   
    int SegSize = num_elements/2;
    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);
  
    cuda_ret = cudaMalloc((void**)&in_d1, SegSize * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) printf("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&bins_d1, num_bins * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) printf("Unable to allocate device memory");
	
    cuda_ret = cudaMalloc((void**)&in_d2, SegSize * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) printf("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&bins_d2, num_bins * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) printf("Unable to allocate device memory");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    //cuda_ret = cudaMemcpy(in_d, in_h, num_elements * sizeof(unsigned int),
        cudaMemcpyHostToDevice);
    //if(cuda_ret != cudaSuccess) printf("Unable to copy memory to the device");

    cuda_ret = cudaMemset(bins_d, 0, num_bins * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) printf("Unable to set device memory");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel ----------------------------------------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);
	
    dim3 block(BLOCK_SIZE);
    dim3 grid((BLOCK_SIZE + block.x - 1)/block.x);
    for (int i = 0; i <num_elements ; i+=SegSize*2) {
	cudaMemcpyAsync(in_d1, in_h+i, SegSize * sizeof(unsigned int),cudaMemcpyHostToDevice,stream0);
	cudaMemcpyAsync(in_d2, in_h+i+SegSize, SegSize * sizeof(unsigned int),cudaMemcpyHostToDevice,stream1);
	histo_kernel<<<grid,block,0,stream0>>>(in_d1,bins_d1,SegSize,num_bins);
	histo_kernel<<<grid,block,0,stream1>>>(in_d2,bins_d2,SegSize,num_bins);
	cudaMemcpy(bins_h+i, bins_d1, num_bins * sizeof(unsigned int),cudaMemcpyDeviceToHost,stream0);
	cudaMemcpy(bins_h+i+SegSize, bins_d2, num_bins * sizeof(unsigned int),cudaMemcpyDeviceToHost,stream1);
    }
	    
    //histogram(in_d, bins_d, num_elements, num_bins);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) printf("Unable to launch/execute kernel");

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    //cuda_ret = cudaMemcpy(bins_h, bins_d, num_bins * sizeof(unsigned int),
        //cudaMemcpyDeviceToHost);
	  //if(cuda_ret != cudaSuccess) printf("Unable to copy memory to host");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify(in_h, bins_h, num_elements, num_bins);

    // Free memory ------------------------------------------------------------

    cudaFree(in_d1); cudaFree(bins_d1);
    cudaFree(in_d2); cudaFree(bins_d2);
    free(in_h); free(bins_h);

    return 0;
}


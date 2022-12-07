#include <stdio.h>
#define BLOCK_SIZE 512
__global__ void histo_kernel(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins)
{
	
    /*************************************************************************/
    // INSERT KERNEL CODE HERE

        int tid=threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ unsigned int bin[4096];
	unsigned int binIdx = threadIdx.x;
	int offset= blockDim.x * gridDim.x;
	while (binIdx<num_bins)
	{
		bin[binIdx] = 0;
		binIdx += blockDim.x;
	}
	__syncthreads();
	
	while(tid<num_elements)
	{
		atomicAdd(&(bin[(unsigned int)input[tid]]), 1); 
		tid+=offset;
	}
  	__syncthreads();
	for (unsigned int binIdx = threadIdx.x; binIdx < num_bins; binIdx += blockDim.x) 
	{
		atomicAdd(&(bins[binIdx]), bin[binIdx]);
	}
}	
	  /*************************************************************************/

void histogram(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins) {

	  /*************************************************************************/
    //INSERT CODE HERE


	dim3 block(BLOCK_SIZE);
	dim3 grid((BLOCK_SIZE + block.x - 1)/block.x);
	histo_kernel<<<grid,block>>>(input,bins,num_elements,num_bins);

	  /*************************************************************************/

}



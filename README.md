# cudahistogram
To create and run the histogram program using streams, you can follow these steps:

1. Clone the repository containing the source code by running the command **`git clone`**. This will download the code to your local machine.

2. Navigate to the directory containing the source code and run the **`make`** command to compile the code. This will create the compiled files that are needed to run the program.

3. To run the program, use the command **`./histogram`**. By default, this will run the program with **num_elements = 1000000 and num_bins = 4096**, and will verify the result.

4. To measure the performance of different optimizations, you can use the command **`nvprof ./histogram`**. This will run the program and display performance metrics for each optimization.

Note: Make sure you have the necessary dependencies installed on your system, such as a compatible compiler and the NVIDIA profiling tools (if using the **`nvprof`** command)


**Results/Comaprison:**

**Stream-implemented histogram code:**
<img width="1094" alt="Screenshot 2022-12-08 at 1 16 08 AM" src="https://user-images.githubusercontent.com/57623274/206806957-fee0b9b3-70aa-4a3e-80ae-1b34346275b3.png">

**Histogram code without stream:**
<img width="1094" alt="Screenshot 2022-12-08 at 1 15 26 AM" src="https://user-images.githubusercontent.com/57623274/206807002-5876e47a-f514-4898-86b7-f7feb872d50e.png">


**Conclusion:**
Using CUDA streams on the GPU can potentially improve the performance of an application by allowing it to achieve an additional 2 to 2.5 times more speedup. CUDA streams provide a way to execute multiple independent tasks on the GPU concurrently. This allows for a higher degree of parallelism and can improve performance by allowing the GPU to process more instructions in parallel. However, it is important to note that the actual speedup achieved by using CUDA streams will depend on the specific application and the characteristics of the input data. The optimal number of streams and the performance benefits of using streams may vary depending on these factors.


**Future work:**
In my current implementation, I am using only two streams for data transfer and kernel processing. By increasing the number of streams, I can potentially accelerate my application and improve its performance.
In addition to increasing the number of streams, I can also use compiler directives such as OpenACC to easily port my code to the GPU for acceleration. OpenACC allows me to specify which parts of my code should be executed on the GPU, without the need to explicitly write GPU-specific code.
I can also consider using NVIDIA math libraries, which provide optimized implementations of common mathematical functions. These libraries can help improve the performance of my application by using highly optimized and tuned functions that are optimized for NVIDIA GPUs.
Overall, by using a combination of these techniques, I can potentially improve the performance of my application and accelerate its execution on the GPU.

# cudahistogram
To create and run the histogram program using streams, you can follow these steps:

1. Clone the repository containing the source code by running the command **`git clone`**. This will download the code to your local machine.

2. Navigate to the directory containing the source code and run the **`make`** command to compile the code. This will create the compiled files that are needed to run the program.

3. To run the program, use the command **`./histogram`**. By default, this will run the program with **num_elements = 1000000 and num_bins = 4096**, and will verify the result.

4. To measure the performance of different optimizations, you can use the command **`nvprof ./histogram`**. This will run the program and display performance metrics for each optimization.

Note: Make sure you have the necessary dependencies installed on your system, such as a compatible compiler and the NVIDIA profiling tools (if using the **`nvprof`** command)

gcc -O3 -Wall -g -c -fopenmp lib/c_taillard.c -o lib/c_taillard.o -I/share/compilers/nvidia/cuda/12.0/include -I/usr/local/cuda-11.2/targets/x86_64-linux/include/
gcc -O3 -Wall -g -c -fopenmp lib/c_bound_simple.c -o lib/c_bound_simple.o -I/share/compilers/nvidia/cuda/12.0/include -I/usr/local/cuda-11.2/targets/x86_64-linux/include/
gcc -O3 -Wall -g -c -fopenmp lib/c_bound_johnson.c -o lib/c_bound_johnson.o -I/share/compilers/nvidia/cuda/12.0/include -I/usr/local/cuda-11.2/targets/x86_64-linux/include/
gcc -O3 -Wall -g -c -fopenmp lib/PFSP_node.c -o lib/PFSP_node.o -I/share/compilers/nvidia/cuda/12.0/include -I/usr/local/cuda-11.2/targets/x86_64-linux/include/
gcc -O3 -Wall -g -c -fopenmp lib/Auxiliary.c -o lib/Auxiliary.o -I/share/compilers/nvidia/cuda/12.0/include -I/usr/local/cuda-11.2/targets/x86_64-linux/include/
gcc -O3 -Wall -g -c -fopenmp lib/Pool.c -o lib/Pool.o -I/share/compilers/nvidia/cuda/12.0/include -I/usr/local/cuda-11.2/targets/x86_64-linux/include/
hipify-perl lib/evaluate.cu > lib/evaluate.cu.hip
hipcc -O3 -offload-arch=gfx906 lib/evaluate.cu.hip -o evaluate_hip.o -lm -L/opt/rocm-4.5.0/hip/lib
hipify-perl lib/c_bounds_gpu.cu > lib/c_bounds_gpu.cu.hip
hipcc -O3 -offload-arch=gfx906 lib/c_bounds_gpu.cu.hip -o c_bounds_gpu_hip.o -lm -L/opt/rocm-4.5.0/hip/lib
hipify-perl pfsp_gpu_cuda.c > pfsp_gpu_hip.c.hip
gcc -O3 -Wall -g -c -fopenmp pfsp_gpu_hip.c.hip -o pfsp_gpu_hip.o -I/share/compilers/nvidia/cuda/12.0/include -I/usr/local/cuda-11.2/targets/x86_64-linux/include/ -lm -L/opt/rocm-4.5.0/hip/lib
gcc -O3 -Wall -g pfsp_gpu_hip.o lib/c_taillard.o lib/c_bound_simple.o lib/c_bound_johnson.o lib/PFSP_node.o lib/Auxiliary.o lib/evaluate_hip.o lib/c_bounds_gpu_hip.o lib/Pool.o -o pfsp_gpu_hip.out -lm -lcudart -L/usr/local/cuda-11.2/targets/x86_64-linux/lib/
#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include "evaluate.h"
#include "c_bounds_gpu.cu"
  
  __device__ void swap_cuda(int* a, int* b)
  {
    int tmp = *b;
    *b = *a;
    *a = tmp;
  }
  
  void printDims(dim3 gridDim, dim3 blockDim) {
    printf("Grid Dimensions : [%d, %d, %d] blocks. \n",
	   gridDim.x, gridDim.y, gridDim.z);
    
    printf("Block Dimensions : [%d, %d, %d] threads.\n",
	   blockDim.x, blockDim.y, blockDim.z);
  }

  // Evaluate a bulk of parent nodes on GPU using lb1
  __global__ void evaluate_gpu_lb1 (const int jobs, const int size, Node* parents_d, const lb1_bound_data  lbound1_d, int* nb_jobs_d, int* nb_machines_d, int* bounds)
  {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int nb_jobs = *nb_jobs_d;
    int nb_machines = *nb_machines_d;
    
    if (threadId < size) {
      const int parentId = threadId / jobs; 
      const int k = threadId % jobs; 
      Node parent =  parents_d[parentId];
      int depth = parent.depth;
      int limit1 = parent.limit1;

      // We evaluate all permutations by varying index k from limit1 forward
      if (k >= limit1+1) {
	swap_cuda(&parent.prmu[depth],&parent.prmu[k]);
	lb1_bound_gpu(lbound1_d, nb_jobs, nb_machines, parent.prmu, limit1+1, jobs, &bounds[threadId]);
	//swap_cuda(&parent.prmu[depth],&parent.prmu[k]);
      }
    }
  }

  /*
    NOTE: This lower bound evaluates all the children of a given parent at the same time.
    Therefore, the GPU loop is on the parent nodes and not on the children ones, in contrast
    to the other lower bounds.
  */
  // Evaluate a bulk of parent nodes on GPU using lb1_d.
  __global__ void evaluate_gpu_lb1_d(const int jobs, const int size, Node* parents_d, const lb1_bound_data lbound1_d, int nb_jobs_d, int nb_machines_d, int* bounds)
  {
    int parentId = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(parentId < size){ 
      Node parent = parents_d[parentId];
     
      // Vector of integers of size MAX_JOBS
      int lb_begin[MAX_JOBS];
    
      lb1_children_bounds_gpu(lbound1_d, nb_jobs_d, nb_machines_d, parent.prmu, parent.limit1, jobs, lb_begin);

      for(int k = 0; k < jobs; k++) {
	if (k >= parent.limit1+1) {
	  const int job = parent.prmu[k];
	  bounds[parentId*jobs+k] = lb_begin[job];
	}
      }
    }
  }

  // Evaluate a bulk of parent nodes on GPU using lb2.
  __global__ void evaluate_gpu_lb2(const int jobs, const int size, int best, Node* parents_d, const lb1_bound_data lbound1_d, const lb2_bound_data lbound2_d, int* nb_jobs_d, int* nb_machines_d, int* bounds)
  {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int nb_jobs = *nb_jobs_d;
    int nb_machines = *nb_machines_d;

    if (threadId < size) {
      const int parentId = threadId / jobs; 
      const int k = threadId % jobs; 
      Node parent =  parents_d[parentId];
      int depth = parent.depth;
      int limit1 = parent.limit1;

      // We evaluate all permutations by varying index k from limit1 forward
      if (k >= limit1+1) {
	swap_cuda(&parent.prmu[depth],&parent.prmu[k]);
	lb2_bound_gpu(lbound1_d, lbound2_d, nb_jobs, nb_machines, parent.prmu, limit1+1, jobs, best, &bounds[threadId]);
	//swap_cuda(&parent.prmu[depth],&parent.prmu[k]);
      }
    }
  } 
 

  void evaluate_gpu(const int jobs, const int lb, const int size, const int nbBlocks, const int nbBlocks_lb1_d, int* best, const lb1_bound_data lbound1, int* nb_jobs_d, int* nb_machines_d, const lb2_bound_data lbound2, Node* parents, int* bounds)
  {
    // 1D grid of 1D nbBlocks(_lb1_d) blocks with block size BLOCK_SIZE
    switch (lb) {
    case 0: // lb1_d
      evaluate_gpu_lb1_d<<<nbBlocks_lb1_d, BLOCK_SIZE>>>(jobs, size, parents, lbound1, *nb_jobs_d, *nb_machines_d, bounds);
      return;
      break;

    case 1: // lb1
      evaluate_gpu_lb1<<<nbBlocks, BLOCK_SIZE>>>(jobs, size, parents, lbound1, nb_jobs_d, nb_machines_d, bounds);
      return;
      break;

    case 2: // lb2
      evaluate_gpu_lb2<<<nbBlocks, BLOCK_SIZE>>>(jobs, size, *best, parents, lbound1, lbound2, nb_jobs_d, nb_machines_d, bounds);
      return;
      break;
    }
  }

  __global__ void print_info_cuda(const lb1_bound_data* lbound1_d)
  {
    // printf("Arrived here\n");
    // printf("For lbound1_d data:\n lbound1_d->jobs = %d, lbound1_d->nb_machines = %d\n",lbound1_d->nb_jobs, lbound1_d->nb_machines);
    // printf("p_times: ");
    // for(int i = 0; i < lbound1_d->nb_jobs*lbound1_d->nb_machines; i++){
    //   printf("%d ", lbound1_d->p_times[i]);
    // }
    // printf("\n");

    // printf("min_heads: ");
    // for(int i = 0; i < lbound1_d->nb_machines; i++){
    //   printf("%d ", lbound1_d->min_heads[i]);
    // }
    // printf("\n");
    
    // printf("min_tails: ");
    // for(int i = 0; i < lbound1_d->nb_machines; i++){
    //   printf("%d ", lbound1_d->min_tails[i]);
    // }
    // printf("\n");

    int sum = 0;
    
    for(int i = 0; i < lbound1_d->nb_jobs*lbound1_d->nb_machines; i++){
      sum += lbound1_d->p_times[i];
    }
    
    for(int i = 0; i < lbound1_d->nb_machines; i++){
      sum += lbound1_d->min_heads[i];
    }
    
    for(int i = 0; i < lbound1_d->nb_machines; i++){
      sum += lbound1_d->min_tails[i];
    }
   
    return;
  }

   __global__ void print_info_cuda2(const lb2_bound_data* lbound2)
  {
    // printf("Arrived here\n");
    // printf("For lbound1_d data:\n lbound1_d->jobs = %d, lbound1_d->nb_machines = %d\n",lbound1_d->nb_jobs, lbound1_d->nb_machines);
    // printf("p_times: ");
    // for(int i = 0; i < lbound1_d->nb_jobs*lbound1_d->nb_machines; i++){
    //   printf("%d ", lbound1_d->p_times[i]);
    // }
    // printf("\n");

    // printf("min_heads: ");
    // for(int i = 0; i < lbound1_d->nb_machines; i++){
    //   printf("%d ", lbound1_d->min_heads[i]);
    // }
    // printf("\n");
    
    // printf("min_tails: ");
    // for(int i = 0; i < lbound1_d->nb_machines; i++){
    //   printf("%d ", lbound1_d->min_tails[i]);
    // }
    // printf("\n");

    int sum = 0;
    
    for(int i = 0; i < lbound2->nb_machine_pairs; i++){
      sum += lbound2->lags[i];
    }
    
    for(int i = 0; i < lbound2->nb_machine_pairs; i++){
      sum += lbound2->machine_pairs_1[i];
    }
    
    for(int i = 0; i < lbound2->nb_machine_pairs; i++){
      sum += lbound2->machine_pairs_2[i];
    }
   
    return;
  }

     __global__ void print_info_cuda2_d(const lb2_bound_data lbound2)
  {
    // printf("Arrived here\n");
    // printf("For lbound1_d data:\n lbound1_d->jobs = %d, lbound1_d->nb_machines = %d\n",lbound1_d->nb_jobs, lbound1_d->nb_machines);
    // printf("p_times: ");
    // for(int i = 0; i < lbound1_d->nb_jobs*lbound1_d->nb_machines; i++){
    //   printf("%d ", lbound1_d->p_times[i]);
    // }
    // printf("\n");

    // printf("min_heads: ");
    // for(int i = 0; i < lbound1_d->nb_machines; i++){
    //   printf("%d ", lbound1_d->min_heads[i]);
    // }
    // printf("\n");
    
    // printf("min_tails: ");
    // for(int i = 0; i < lbound1_d->nb_machines; i++){
    //   printf("%d ", lbound1_d->min_tails[i]);
    // }
    // printf("\n");

    int sum = 0;
    
    for(int i = 0; i < lbound2.nb_machine_pairs; i++){
      sum += lbound2.lags[i];
    }
    
    for(int i = 0; i < lbound2.nb_machine_pairs; i++){
      sum += lbound2.machine_pairs_1[i];
    }
    
    for(int i = 0; i < lbound2.nb_machine_pairs; i++){
      sum += lbound2.machine_pairs_2[i];
    }
   
    return;
  }
  
  void print_info(const lb1_bound_data* lbound1_d)
  {
    print_info_cuda<<<200,512>>>(lbound1_d);
    
    return;
  }

  void print_info_lb2(const lb2_bound_data* lbound2)
  {
    print_info_cuda2<<<200,512>>>(lbound2);
  }

  void print_info_lb2_d(const lb2_bound_data lbound2_d){

    print_info_cuda2_d<<<200,512>>>(lbound2_d);
  }
  
#ifdef __cplusplus
}
#endif

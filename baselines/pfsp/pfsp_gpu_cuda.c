/*
  Single CUDA GPU B&B to solve Taillard instances of the PFSP in C.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <getopt.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

//#include "parameters.h"
#include "lib/c_bound_simple.h"
#include "lib/c_bound_johnson.h"
#include "lib/c_taillard.h"
#include "evaluate.h"


//#define BLOCK_SIZE 512

/*******************************************************************************
Implementation of PFSP Nodes.
*******************************************************************************/

/* #define MAX_JOBS 20 */

/* typedef struct */
/* { */
/*   uint8_t depth; */
/*   int limit1; */
/*   int prmu[MAX_JOBS]; */
/* } Node; */

// Initialization of nodes is done by CPU only

void initRoot(Node* root, const int jobs)
{
  root->depth = 0;
  root->limit1 = -1;
  for (int i = 0; i < jobs; i++) {
    root->prmu[i] = i;
  }
}

/*******************************************************************************
Implementation of a dynamic-sized single pool data structure.
Its initial capacity is 1024, and we reallocate a new container with double
the capacity when it is full. Since we perform only DFS, it only supports
'pushBack' and 'popBack' operations.
*******************************************************************************/

// Pools are managed by the CPU only

#define CAPACITY 1024

typedef struct
{
  Node* elements;
  int capacity;
  int size;
} SinglePool;

void initSinglePool(SinglePool* pool)
{
  pool->elements = (Node*)malloc(CAPACITY * sizeof(Node));
  pool->capacity = CAPACITY;
  pool->size = 0;
}

void pushBack(SinglePool* pool, Node node)
{
  if (pool->size >= pool->capacity) {
    pool->capacity *= 2;
    pool->elements = (Node*)realloc(pool->elements, pool->capacity * sizeof(Node));
  }

  pool->elements[pool->size++] = node;
}

Node popBack(SinglePool* pool, int* hasWork)
{
  if (pool->size > 0) {
    *hasWork = 1;
    return pool->elements[--pool->size];
  }

  return (Node){0};
}

void deleteSinglePool(SinglePool* pool)
{
  free(pool->elements);
}

/*******************************************************************************
Implementation of the parallel CUDA GPU PFSP search.
*******************************************************************************/

void parse_parameters(int argc, char* argv[], int* inst, int* lb, int* ub, int* m, int *M)
{
  *m = 25;
  *M = 50000;
  *inst = 14;
  *lb = 1;
  *ub = 1;
  /*
    NOTE: Only forward branching is considered because other strategies increase a
    lot the implementation complexity and do not add much contribution.
  */

  // Define long options
  static struct option long_options[] = {
    {"inst", required_argument, NULL, 'i'},
    {"lb", required_argument, NULL, 'l'},
    {"ub", required_argument, NULL, 'u'},
    {"m", required_argument, NULL, 'm'},
    {"M", required_argument, NULL, 'M'},
    {NULL, 0, NULL, 0} // Terminate options array
  };

  int opt, value;
  int option_index = 0;

  while ((opt = getopt_long(argc, argv, "i:l:u:m:M", long_options, &option_index)) != -1) {
    value = atoi(optarg);

    switch (opt) {
    case 'i':
      if (value < 1 || value > 120) {
	fprintf(stderr, "Error: unsupported Taillard's instance\n");
	exit(EXIT_FAILURE);
      }
      *inst = value;
      break;

    case 'l':
      if (value < 0 || value > 2) {
	fprintf(stderr, "Error: unsupported lower bound function\n");
	exit(EXIT_FAILURE);
      }
      *lb = value;
      break;

    case 'u':
      if (value != 0 && value != 1) {
	fprintf(stderr, "Error: unsupported upper bound initialization\n");
	exit(EXIT_FAILURE);
      }
      *ub = value;
      break;

    case 'm':
      if (value < 25 || value > 100) {
	fprintf(stderr, "Error: unsupported minimal pool for GPU initialization\n");
	exit(EXIT_FAILURE);
      }
      *m = value;
      break;

    case 'M':
      if (value < 45000 || value > 50000) {
	fprintf(stderr, "Error: unsupported maximal pool for GPU initialization\n");
	exit(EXIT_FAILURE);
      }
      *M = value;
      break;

    default:
      fprintf(stderr, "Usage: %s --inst <value> --lb <value> --ub <value> --m <value> --M <value>\n", argv[0]);
      exit(EXIT_FAILURE);
    }
  }
}

void print_settings(const int inst, const int machines, const int jobs, const int ub, const int lb)
{
  printf("\n=================================================\n");
  printf("Parallel GPU CUDA\n\n");
  printf("Resolution of PFSP Taillard's instance: ta%d (m = %d, n = %d) using parallel GPU CUDA\n", inst, machines, jobs);
  if (ub == 0) printf("Initial upper bound: inf\n");
  else /* if (ub == 1) */ printf("Initial upper bound: opt\n");
  if (lb == 0) printf("Lower bound function: lb1_d\n");
  else if (lb == 1) printf("Lower bound function: lb1\n");
  else /* (lb == 2) */ printf("Lower bound function: lb2\n");
  printf("Branching rule: fwd\n");
  printf("=================================================\n");
}

void print_results(const int optimum, const unsigned long long int exploredTree,
		   const unsigned long long int exploredSol, const double timer)
{
  printf("\n=================================================\n");
  printf("Size of the explored tree: %llu\n", exploredTree);
  printf("Number of explored solutions: %llu\n", exploredSol);
  /* TODO: Add 'is_better' */
  printf("Optimal makespan: %d\n", optimum);
  printf("Elapsed time: %.4f [s]\n", timer);
  printf("=================================================\n");
}

inline void swap(int* a, int* b)
{
  int tmp = *b;
  *b = *a;
  *a = tmp;
}

// Evaluate and generate children nodes on CPU.
void decompose_lb1(const int jobs, const lb1_bound_data* const lbound1, const Node parent,
		   int* best, unsigned long long int* tree_loc, unsigned long long int* num_sol, SinglePool* pool)
{
  for (int i = parent.limit1+1; i < jobs; i++) {
    Node child;
    memcpy(child.prmu, parent.prmu, jobs * sizeof(int));
    swap(&child.prmu[parent.depth], &child.prmu[i]);
    child.depth = parent.depth + 1;
    child.limit1 = parent.limit1 + 1;

    int lowerbound = lb1_bound(lbound1, child.prmu, child.limit1, jobs);

    if (child.depth == jobs) { // if child leaf
      *num_sol += 1;

      if (lowerbound < *best) { // if child feasible
        *best = lowerbound;
      }
    } else { // if not leaf
      if (lowerbound < *best) { // if child feasible
        pushBack(pool, child);
        *tree_loc += 1;
      }
    }
  }
}

void decompose_lb1_d(const int jobs, const lb1_bound_data* const lbound1, const Node parent,
		     int* best, unsigned long long int* tree_loc, unsigned long long int* num_sol, SinglePool* pool)
{
  int* lb_begin = (int*)malloc(jobs * sizeof(int));

  lb1_children_bounds(lbound1, parent.prmu, parent.limit1, jobs, lb_begin);

  for (int i = parent.limit1+1; i < jobs; i++) {
    const int job = parent.prmu[i];
    const int lb = lb_begin[job];

    if (parent.depth + 1 == jobs) { // if child leaf
      *num_sol += 1;

      if (lb < *best) { // if child feasible
        *best = lb;
      }
    } else { // if not leaf
      if (lb < *best) { // if child feasible
        Node child;
        memcpy(child.prmu, parent.prmu, jobs * sizeof(int));
        child.depth = parent.depth + 1;
        child.limit1 = parent.limit1 + 1;
        swap(&child.prmu[child.limit1], &child.prmu[i]);

        pushBack(pool, child);
        *tree_loc += 1;
      }
    }
  }

  free(lb_begin);
}

void decompose_lb2(const int jobs, const lb1_bound_data* const lbound1, const lb2_bound_data* const lbound2,
		   const Node parent, int* best, unsigned long long int* tree_loc, unsigned long long int* num_sol,
		   SinglePool* pool)
{
  for (int i = parent.limit1+1; i < jobs; i++) {
    Node child;
    memcpy(child.prmu, parent.prmu, jobs * sizeof(int));
    swap(&child.prmu[parent.depth], &child.prmu[i]);
    child.depth = parent.depth + 1;
    child.limit1 = parent.limit1 + 1;

    int lowerbound = lb2_bound(lbound1, lbound2, child.prmu, child.limit1, jobs, *best);

    if (child.depth == jobs) { // if child leaf
      *num_sol += 1;

      if (lowerbound < *best) { // if child feasible
        *best = lowerbound;
      }
    } else { // if not leaf
      if (lowerbound < *best) { // if child feasible
        pushBack(pool, child);
        *tree_loc += 1;
      }
    }
  }
}

void decompose(const int jobs, const int lb, int* best,
	       const lb1_bound_data* const lbound1, const lb2_bound_data* const lbound2, const Node parent,
	       unsigned long long int* tree_loc, unsigned long long int* num_sol, SinglePool* pool)
{
  switch (lb) {
  case 0: // lb1_d
    decompose_lb1_d(jobs, lbound1, parent, best, tree_loc, num_sol, pool);
    break;

  case 1: // lb1
    decompose_lb1(jobs, lbound1, parent, best, tree_loc, num_sol, pool);
    break;

  case 2: // lb2
    decompose_lb2(jobs, lbound1, lbound2, parent, best, tree_loc, num_sol, pool);
    break;
  }
}

// Generate children nodes (evaluated on GPU) on CPU
void generate_children(Node* parents, const int size, const int jobs, int* bounds,
		       unsigned long long int* exploredTree, unsigned long long int* exploredSol, int* best, SinglePool* pool)
{
  for (int i = 0; i < size; i++) {
    Node parent = parents[i];
    const uint8_t depth = parent.depth;

    for (int j = parent.limit1+1; j < jobs; j++) {
      const int lowerbound = bounds[j + i * jobs];

      // If child leaf
      if(depth + 1 == jobs){
	exploredSol += 1;

	// If child feasible
	if(lowerbound < *best) *best = lowerbound;

      } else { // If not leaf
	if(lowerbound < *best) {
	  Node child;
	  memcpy(child.prmu, parent.prmu, jobs * sizeof(int));
	  swap(&child.prmu[parent.depth], &child.prmu[j]);
	  child.depth = parent.depth + 1;
	  child.limit1 = parent.limit1 + 1;
	  
	  pushBack(pool, child);
	  exploredTree += 1;
	}
      }
    }
  }
}

// Single-GPU PFSP search
void pfsp_search(const int inst, const int lb, const int m, const int M, int* best,
		 unsigned long long int* exploredTree, unsigned long long int* exploredSol,
		 double* elapsedTime)
{
  // Initializing problem
  int jobs = taillard_get_nb_jobs(inst);
  int machines = taillard_get_nb_machines(inst);

  printf("%d number of jobs and %d number of machines", jobs, machines);
  
  // Starting pool
  Node root;
  initRoot(&root, jobs);

  SinglePool pool;
  initSinglePool(&pool);

  pushBack(&pool, root);

  // Timer
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);

  lb1_bound_data* lbound1;
  lbound1 = new_bound_data(jobs, machines);
  taillard_get_processing_times(lbound1->p_times, inst);
  fill_min_heads_tails(lbound1);
    
  lb2_bound_data* lbound2;
  lbound2 = new_johnson_bd_data(lbound1);
  fill_machine_pairs(lbound2/*, LB2_FULL*/);
  fill_lags(lbound1->p_times, lbound2);
  fill_johnson_schedules(lbound1->p_times, lbound2);

  // Passing lb1 bounding data to GPU
  lb1_bound_data* lbound1_d;
  cudaMalloc((void**)&lbound1_d, sizeof(lb1_bound_data));
  cudaMemcpy((void*)lbound1_d, (void*)lbound1, sizeof(lb1_bound_data), cudaMemcpyHostToDevice);

  // Passing lb2 bounding data to GPU
  lb2_bound_data* lbound2_d;
  cudaMalloc((void**)&lbound2_d, sizeof(lb2_bound_data));
  cudaMemcpy((void*)lbound2_d, (void*)lbound2, sizeof(lb2_bound_data), cudaMemcpyHostToDevice);

  // We allocate both parents and bounds vectors with maximum size (?)
  
  // Allocating parents vectors on CPU and GPU
  Node* parents = (Node*)malloc(M * sizeof(Node));
  Node* parents_d;
  cudaMalloc((void**)&parents_d, M * sizeof(Node));

  // Allocating bounds vector on CPU and GPU
  int* bounds = (int*)malloc((jobs*M) * sizeof(int));
  int *bounds_d;
  cudaError_t err = cudaMalloc((void**)&bounds_d, (jobs*M) * sizeof(int));

  if (err != cudaSuccess) {
    printf("Failed to allocate memory on the GPU");
    return;
  }
  
  cudaDeviceSynchronize();

  while (1) {
    int hasWork = 0;
    Node parent = popBack(&pool, &hasWork);
    if (!hasWork) break;

    decompose(jobs, lb, best, lbound1, lbound2, parent, exploredTree, exploredSol, &pool);

    //printf("Size of pool.size = %d\n", pool.size);
    int poolSize = MIN(pool.size,M);

    // If 'poolSize' is sufficiently large, we offload the pool on GPU.
    if (poolSize >= m) {
      
      for(int i= 0; i < poolSize; i++) {
	int hasWork = 0;
	parents[i] = popBack(&pool,&hasWork); //parents size is good because pool is max equals to M
	if (!hasWork) break;
      }
	
      /*
	TODO: Optimize 'numBounds' based on the fact that the maximum number of
	generated children for a parent is 'parent.limit2 - parent.limit1 + 1' or
	something like that.
      */
      const int  numBounds = jobs * poolSize;   
      const int nbBlocks = ceil((double)numBounds / BLOCK_SIZE);
      //printf("numBounds: %d\n nbBlocks size: %d\n", numBounds,nbBlocks);
      cudaMemcpy((void*)parents_d, (void*)parents, poolSize * sizeof(Node), cudaMemcpyHostToDevice); //size of copy is good
      cudaDeviceSynchronize();

      // count += 1;
      // numBounds is the 'size' of the problem
      evaluate_gpu(jobs, lb, numBounds, nbBlocks, best, lbound1_d, lbound2_d, parents_d, bounds_d); 
      //printf("Value of 10th position of bounds_d = %d", bounds_d[9]);
      cudaDeviceSynchronize();
      
      cudaMemcpy((int*)bounds, (int*)bounds_d, numBounds * sizeof(int), cudaMemcpyDeviceToHost); //size of copy is good

      cudaDeviceSynchronize();
      
      /*
	Each task generates and inserts its children nodes to the pool.
      */
      generate_children(parents, poolSize, jobs, bounds, exploredTree, exploredSol, best, &pool);
    }
  }

  // Attention to these free data
  // labels are represented by the bounds now and they have their own freeing functions (see below)
  cudaFree(parents_d);
  cudaFree(bounds_d);
  cudaFree(lbound1_d);
  cudaFree(lbound2_d);
  
  free(parents);
  free(bounds);
        
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  *elapsedTime = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  printf("\nExploration terminated.\n");

  // Freeing memory
  deleteSinglePool(&pool);
  free_bound_data(lbound1);
  free_johnson_bd_data(lbound2);
}

int main(int argc, char* argv[])
{
  int inst, lb, ub, m, M;
  parse_parameters(argc, argv, &inst, &lb, &ub, &m, &M);

  int jobs = taillard_get_nb_jobs(inst);
  int machines = taillard_get_nb_machines(inst);

  print_settings(inst, machines, jobs, ub, lb);

  int optimum = (ub == 1) ? taillard_get_best_ub(inst) : INT_MAX;
  unsigned long long int exploredTree = 0;
  unsigned long long int exploredSol = 0;

  double elapsedTime;

  pfsp_search(inst, lb, m, M, &optimum, &exploredTree, &exploredSol, &elapsedTime);

  print_results(optimum, exploredTree, exploredSol, elapsedTime);

  printf("We are done\n");

  return 0;
}

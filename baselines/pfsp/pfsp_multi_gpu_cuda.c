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
#include <omp.h>
#include <stdbool.h>
//#include <cuda.h>
#include <cuda_runtime.h>

#include "lib/c_bound_simple.h"
#include "lib/c_bound_johnson.h"
#include "lib/c_taillard.h"
#include "evaluate.h"


/*******************************************************************************
Implementation of PFSP Nodes.
*******************************************************************************/

// BLOCK_SIZE, MAX_JOBS and struct Node are defined in parameters.h

// Initialization of nodes is done by CPU only

void initRoot(Node* root, const int jobs)
{
  root->depth = 0;
  root->limit1 = -1;
  for (int i = 0; i < jobs; i++) {
    root->prmu[i] = i;
  }
}

// Pools are managed by the CPU only

/*******************************************************************************
  Implementation of a dynamic-sized single pool data structure.
  Its initial capacity is 1024, and we reallocate a new container with double
  the capacity when it is full. Since we perform only DFS, it only supports
  'pushBack' and 'popBack' operations.
*******************************************************************************/

#define CAPACITY 1024

typedef struct
{
  Node* elements;
  int capacity;
  int front;
  int size;
  bool lock;
} SinglePool_ext;

void initSinglePool(SinglePool* pool)
{
  pool->elements = (Node*)malloc(CAPACITY * sizeof(Node));
  pool->capacity = CAPACITY;
  pool->front = 0;
  pool->size = 0;
  pool->lock = false;
}

void pushBack(SinglePool_ext* pool, Node* node) {
  while (true) {
    if (__sync_bool_compare_and_swap(&(pool->lock), false, true)) {
      if (pool->front + pool->size >= pool->capacity) {
	pool->capacity *= 2;
	pool->elements = realloc(pool->elements, pool->capacity * sizeof(Node));
      }

      // Copy node to the end of elements array
      memcpy(pool->elements[(pool->front + pool->size)], node, sizeof(Node));
      pool->size += 1;
      pool->lock = false;
      return;
    }

    // Yield execution (use appropriate synchronization in actual implementation)
  }
}

void pushBackBulk(SinglePool_ext* pool, SinglePool_ext* nodes) {
  int s = nodes->size;
  while (true) {
    if (__sync_bool_compare_and_swap(&(pool->lock), false, true)) {
      if (pool->front + pool->size >= pool->capacity) {
	pool->capacity *= 2;
	pool->elements = realloc(pool->elements, pool->capacity * sizeof(Node));
      }
      
      // Copy of elements from nodes to the end of elements array of pool
      for(int i = 0; i < s; i++)
	memcpy(pool->elements[(pool->front + pool->size)+i], nodes.elements[i], sizeof(Node));
      pool->size += s;
      pool->lock = false;
      return;
    }
    
    // Yield execution (use appropriate synchronization in actual implementation)
  }
}

Node popBack(SinglePool_ext* pool, int* hasWork) {
  while (true) {
    if (__sync_bool_compare_and_swap(&(pool->lock), false, true)) {
      if (pool->size > 0) {
	*hasWork = 1;
	pool->size -= 1;
	// Copy last element to elt
	Node elt;
	memcpy(elt, pool->elements[(pool->front + pool->size)], sizeof(Node));
	pool->lock = false;
	return elt;
      } else {
	pool->lock = false;
	break;
      }
    }
    // Yield execution (use appropriate synchronization in actual implementation)
  }
  return (Node){0};
}

Node popBackFree(SinglePool_ext* pool, int* hasWork) {
  if (pool->size > 0){
    *hasWork = 1;
    pool->size -= 1;
    return pool->elements[pool->front + pool->size];
  }

  return (Node){0};
}

int popBackBulk(SinglePool_ext* pool, const int m, const int M, Node* parents){
  while(true) {
    if (__sync_bool_compare_and_swap(&(pool->lock), false, true)) {
      if (pool->size < m) {
	pool->lock = false;
	break;
      }
      else{
	int poolSize = MIN(pool->size,M);
	pool->size -= poolSize;
	for(int i = 0; i < poolSize; i++)
	  memcpy(parents[i], pool->elements[(pool->front + pool->size)+i], sizeof(Node));
	pool->lock = false;
	return poolSize;
      }
    }
    // Yield execution (use appropriate synchronization in actual implementation)
  }
  return 0;
}

Node* popBackBulkFree(SinglePool_ext* pool, const int m, const int M, int* poolSize){
  if(pool.size >= 2*m) {
    *poolSize = pool.size/2;
    pool.size -= *poolSize;
    Node* parents = (Node*)malloc(*poolSize * sizeof(Node));
    for(int i = 0; i < *poolSize; i++)
      memcpy(parents[i], pool->elements[(pool->front + pool->size)+i], sizeof(Node));
    return parents;
  }else{
    return NULL;
  }
  
  Node* parents = NULL;
  return parents;
}

Node popFront(SinglePool_ext* pool, int* hasWork) {
  if(pool->size > 0) {
    *hasWork = 1;
    Node node;
    memcpy(node,pool->elements[pool.front],sizeof(Node));
    pool->front += 1;
    pool->size -= 1;
    return node;
  }

  return (Node){0};
}

void deleteSinglePool_ext(SinglePool_ext* pool) {
  free(pool->elements);
  pool->elements = NULL;
  pool->capacity = 0;
  pool->front = 0;
  pool->size = 0;
  pool->lock = false;
}

/*******************************************************************************
Implementation of the parallel CUDA GPU PFSP search.
*******************************************************************************/

void parse_parameters(int argc, char* argv[], int* inst, int* lb, int* ub, int* m, int *M, int *D)
{
  *m = 25;
  *M = 50000;
  *inst = 14;
  *lb = 1;
  *ub = 1;
  *D = 1;
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
    {"D", required_argument, NULL, 'D'},
    {NULL, 0, NULL, 0} // Terminate options array
  };

  int opt, value;
  int option_index = 0;

  while ((opt = getopt_long(argc, argv, "i:l:u:m:M:D:", long_options, &option_index)) != -1) {
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

    case 'D':
      if (value < 0 || value > 16) {
	fprintf(stderr, "Error: unsupported number of GPU's\n");
	exit(EXIT_FAILURE);
      }
      *D = value;
      break;

    default:
      fprintf(stderr, "Usage: %s --inst <value> --lb <value> --ub <value> --m <value> --M <value> --D <value>\n", argv[0]);
      exit(EXIT_FAILURE);
    }
  }
}

 
void print_settings(const int inst, const int machines, const int jobs, const int ub, const int lb, const int D)
{
  printf("\n=================================================\n");
  printf("Parallel multi-GPU CUDA with %d GPU's\n\n", D);
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
void decompose_lb1(const int jobs, const lb1_bound_data* const lbound1, const Node parent, int* best, unsigned long long int* tree_loc, unsigned long long int* num_sol, SinglePool_ext* pool)
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
		     int* best, unsigned long long int* tree_loc, unsigned long long int* num_sol, SinglePool_ext* pool)
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

void decompose_lb2(const int jobs, const lb1_bound_data* const lbound1, const lb2_bound_data* const lbound2, const Node parent, int* best, unsigned long long int* tree_loc, unsigned long long int* num_sol, SinglePool_ext* pool)
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

void decompose(const int jobs, const int lb, int* best, const lb1_bound_data* const lbound1, const lb2_bound_data* const lbound2, const Node parent, unsigned long long int* tree_loc, unsigned long long int* num_sol, SinglePool_ext* pool)
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
void generate_children(Node* parents, const int size, const int jobs, int* bounds, unsigned long long int* exploredTree, unsigned long long int* exploredSol, int* best, SinglePool_ext* pool)
{
  for (int i = 0; i < size; i++) {
    Node parent = parents[i];
    const uint8_t depth = parent.depth;

    for (int j = parent.limit1+1; j < jobs; j++) {
      const int lowerbound = bounds[j + i * jobs];

      // If child leaf
      if(depth + 1 == jobs){
	*exploredSol += 1;

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
	  *exploredTree += 1;
	}
      }
    }
  }
}

// Single-GPU PFSP search
void pfsp_search(const int inst, const int lb, const int m, const int M, const int D, int* best,
		 unsigned long long int* exploredTree, unsigned long long int* exploredSol,
		 double* elapsedTime)
{

  // Initializing problem
  int jobs = taillard_get_nb_jobs(inst);
  int machines = taillard_get_nb_machines(inst);
  //int count = 0;
  
  // Starting pool
  Node root;
  initRoot(&root, jobs);

  SinglePool pool;
  initSinglePool(&pool);

  pushBack(&pool, root);

  // Boolean variables for dynamic workload balance
  bool allTasksIdleFlag = false;
  bool eachTaskState[D]; // one task per GPU
  for(int i = 0; i < D; i++)
    eachTaskState[i] = false;

  // Timer
  // struct timespec start, end;
  // clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  double startTime, endTime;
    
  // Bounding data
  lb1_bound_data* lbound1;
  lbound1 = new_bound_data(jobs, machines);
  taillard_get_processing_times(lbound1->p_times, inst);
  fill_min_heads_tails(lbound1);
    
  lb2_bound_data* lbound2;
  lbound2 = new_johnson_bd_data(lbound1);
  fill_machine_pairs(lbound2/*, LB2_FULL*/);
  fill_lags(lbound1->p_times, lbound2);
  fill_johnson_schedules(lbound1->p_times, lbound2);

  // Approach having parents_d as int**
  // parents_h is a table of integers of size M * (MAX_JOBS+2)
  // Each 22 components we have: first 20 for the prmu, 1 for depth and 1 for limit1
  // int *parents_h = (int*) malloc((M*(MAX_JOBS+2))*sizeof(int));
  // Allocation of parents_d on the GPU
  // int *parents_d;
  // cudaMalloc((void**)&parents_d, (M*(MAX_JOBS+2))*sizeof(int));
  
  /*
    Step 1: We perform a partial breadth-first search on CPU in order to create
    a sufficiently large amount of work for GPU computation.
  */
  
  startTime = omp_get_wtime();
  while(pool.size < D*m) {
    // CPU side
    int hasWork = 0;
    Node parent = popFront(&pool, &hasWork);
    if (!hasWork) break;
    
    decompose(jobs, lb, best, lbound1, lbound2, parent, exploredTree, exploredSol, &pool);
  }
  endTime = omp_get_wtime();
  double t1 = endTime - startTime;

  printf("\nInitial search on CPU completed\n");
  printf("Size of the explored tree: %llu\n", *exploredTree);
  printf("Number of explored solutions: %llu\n", *exploredSol);
  printf("Elapsed time: %f [s]\n", t1);


  /*
    Step 2: We continue the search on GPU in a depth-first manner, until there
    is not enough work.
  */

  startTime = omp_get_wtime();
  unsigned long long int eachExploredTree[D], eachExploredSol[D];
  int eachBest[D];
  
  const int poolSize = pool.size;
  const int c = poolSize / D;
  const int l = poolSize - (D-1)*c;
  const int f = pool.front;
  bool lock_p;
  
  pool.front = 0;
  pool.size = 0;

  //var multiPool: [0..#D] SinglePool_ext(Node); //Don't know if this is necessary


#pragma omp parallel for num_threads(D) shared(eachExploredTree, eachExploredSol, eachBest, eachTaskState, pool, lbound1, lbound2)
  for (int gpuID = 0; gpuID < D; gpuID++) {
    cudaSetDevice(gpuID);
    
    // Vectors for deep copy of lbound1 to device
    lb1_bound_data lbound1_d;
    int* p_times_d;
    int* min_heads_d;
    int* min_tails_d;

    // Allocating and copying memory necessary for deep copy of lbound1
    cudaMalloc((void**)&p_times_d, jobs*machines*sizeof(int));
    cudaMalloc((void**)&min_heads_d, machines*sizeof(int));
    cudaMalloc((void**)&min_tails_d, machines*sizeof(int));
    cudaMemcpy(p_times_d, lbound1->p_times, (jobs*machines)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(min_heads_d, lbound1->min_heads, machines*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(min_tails_d, lbound1->min_tails, machines*sizeof(int), cudaMemcpyHostToDevice);

    // Deep copy of lbound1
    lbound1_d.p_times = p_times_d;
    lbound1_d.min_heads = min_heads_d;
    lbound1_d.min_tails = min_tails_d;
    lbound1_d.nb_jobs = lbound1->nb_jobs;
    lbound1_d.nb_machines = lbound1->nb_machines;

    // Vectors for deep copy of lbound2 to device
    lb2_bound_data lbound2_d;
    int *johnson_schedule_d;
    int *lags_d;
    int *machine_pairs_1_d;
    int *machine_pairs_2_d;
    int *machine_pair_order_d;

    // Allocating and copying memory necessary for deep copy of lbound2
    int nb_mac_pairs = lbound2->nb_machine_pairs;
    cudaMalloc((void**)&johnson_schedule_d, (nb_mac_pairs*jobs) * sizeof(int));
    cudaMalloc((void**)&lags_d, (nb_mac_pairs*jobs) * sizeof(int));
    cudaMalloc((void**)&machine_pairs_1_d, nb_mac_pairs * sizeof(int));
    cudaMalloc((void**)&machine_pairs_2_d, nb_mac_pairs * sizeof(int));
    cudaMalloc((void**)&machine_pair_order_d, nb_mac_pairs * sizeof(int));
    cudaMemcpy(johnson_schedule_d, lbound2->johnson_schedules, (nb_mac_pairs*jobs) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(lags_d, lbound2->lags, (nb_mac_pairs*jobs) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(machine_pairs_1_d, lbound2->machine_pairs_1, nb_mac_pairs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(machine_pairs_2_d, lbound2->machine_pairs_2, nb_mac_pairs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(machine_pair_order_d, lbound2->machine_pair_order, nb_mac_pairs * sizeof(int), cudaMemcpyHostToDevice);

    // Deep copy of lbound2
    lbound2_d.johnson_schedules = johnson_schedule_d;
    lbound2_d.lags = lags_d;
    lbound2_d.machine_pairs_1 = machine_pairs_1_d;
    lbound2_d.machine_pairs_2 = machine_pairs_2_d;
    lbound2_d.machine_pair_order = machine_pair_order_d;
    lbound2_d.nb_machine_pairs = lbound2->nb_machine_pairs;
    lbound2_d.nb_jobs = lbound2->nb_jobs;
    lbound2_d.nb_machines = lbound2->nb_machines;

    const int nSteal, nSSteal;
    unsigned long long int tree = 0, sol = 0;
    SinglePool pool_loc;
    initSinglePool(&pool_loc);
    int best_l = best;
    bool taskState = false;

    // each task gets its chunk
    for (int i = 0; i < c; i++) {
      pool_loc.elements[i] = pool.elements[gpuID+f+i*D];
    }
    pool_loc.size += c;
    if (gpuID == D-1) {
      for (int i = c; i < l; i++) {
        pool_loc.elements[i] = pool.elements[(D*c)+f+i-c];
      }
      pool_loc.size += l-c;
    }
    
    // Allocating parents vector on CPU and GPU
    Node* parents = (Node*)malloc(M * sizeof(Node));
    Node* parents_d;
    cudaMalloc((void**)&parents_d, M * sizeof(Node));
    
    // Allocating bounds vector on CPU and GPU
    int* bounds = (int*)malloc((jobs*M) * sizeof(int));
    int *bounds_d;
    cudaMalloc((void**)&bounds_d, (jobs*M) * sizeof(int));
    
      
    while (1) {
      /*
	Each task gets its parents nodes from the pool
      */

      // Static workload balance
      /* int poolSize = pool_loc.size; */
      /* if (poolSize >= m) { */
      /* 	poolSize = MIN(poolSize,M); */
      
      /* 	for(int i= 0; i < poolSize; i++) { */
      /* 	  int hasWork = 0; */
      /* 	  parents[i] = popBack(&pool_loc,&hasWork); */
      /* 	  // Approach with parents_d as int** */
      /* 	  //parents[i] = popBack_p(&pool, &hasWork, parents_h, i);  */
      /* 	  if (!hasWork) break; */
      /* 	} */

      // Dynamic workload bbalance

      int poolSize = popBackBulk(&pool_loc, m, M, parents);


      // HOW TO ADAPT THOOSE IDLE AND BUSY STATES IN C?
      if (poolSize > 0) {
        if (taskState == IDLE) {
          taskState = BUSY;
          eachTaskState[gpuID].write(BUSY);
        }
      
	/*
	  TODO: Optimize 'numBounds' based on the fact that the maximum number of
	  generated children for a parent is 'parent.limit2 - parent.limit1 + 1' or
	  something like that.
	*/
	const int numBounds = jobs * poolSize;   
	const int nbBlocks = ceil((double)numBounds / BLOCK_SIZE);
	const int nbBlocks_lb1_d = ceil((double)nbBlocks/jobs); 

	// Approach with parents_d as int**
	//cudaMemcpy(parents_d, parents_h, (MAX_SIZE) * poolSize * sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(parents_d, parents, poolSize *sizeof(Node), cudaMemcpyHostToDevice);

	// numBounds is the 'size' of the problem
	evaluate_gpu(jobs, lb, numBounds, nbBlocks, nbBlocks_lb1_d, best, lbound1_d, lbound2_d, parents_d, bounds_d/*, front, back, remain*/); 
	cudaDeviceSynchronize();
      
	cudaMemcpy(bounds, bounds_d, numBounds * sizeof(int), cudaMemcpyDeviceToHost); //size of copy is good

	/*
	  each task generates and inserts its children nodes to the pool.
	*/
	generate_children(parents, poolSize, jobs, bounds, &tree, &sol, best, &pool_loc);
      }
      else {
        // work stealing
        int tries = 0;
        bool steal = false;

	// ADAPT PERMUTE
        const victims = permute(0..#D);

        label WS0 while (tries < D && steal == false) {
          const victimID = victims[tries];

          if (victimID != gpuID) { // if not me
            ref victim = multiPool[victimID];
            nSteal += 1;
            var nn = 0;

            label WS1 while (nn < 10) {
              if victim.lock.compareAndSwap(false, true) { // get the lock
		  const size = victim.size;

		  if (size >= 2*m) {
		    var (hasWork, p) = victim.popBackBulkFree(m, M);
		    if (hasWork == 0) {
		      victim.lock.write(false); // reset lock
		      halt("DEADCODE in work stealing");
		    }

		    /* for i in 0..#(size/2) {
		       pool_loc.pushBack(p[i]);
		       } */
		    pool_loc.pushBackBulk(p);

		    steal = true;
		    nSSteal += 1;
		    victim.lock.write(false); // reset lock
		    break WS0;
		  }

		  victim.lock.write(false); // reset lock
		  break WS1;
		}

              nn += 1;
              currentTask.yieldExecution();
            }
          }
          tries += 1;
        }

        if (steal == false) {
          // termination
          if (taskState == BUSY) {
            taskState = IDLE;
            eachTaskState[gpuID].write(IDLE);
          }
          if allIdle(eachTaskState, allTasksIdleFlag) {
	      /* writeln("task ", gpuID, " exits normally"); */
	      break;
	    }
          continue;
        } else {
          continue;
        }
      }
    }

    // OpenMP environment freeing variables
    cudaFree(parents_d);
    cudaFree(bounds_d);
    cudaFree(p_times_d);
    cudaFree(min_heads_d);
    cudaFree(min_tails_d);
    cudaFree(johnson_schedule_d);
    cudaFree(lags_d);
    cudaFree(machine_pairs_1_d);
    cudaFree(machine_pairs_2_d);
    cudaFree(machine_pair_order_d);
    free(parents);
    free(bounds);

    // HERE THERE IS SOMETHING WITH lock_p VARIABLE!
#pragma omp critical
    {
      const int poolLocSize = pool_loc.size;
      for (int i = 0; i < poolLocSize; i++) {
        int hasWork = 0;
        pushBack(&pool, popBack(&pool_loc, &hasWork));
        if (!hasWork) break;
      }
    }

    eachExploredTree[gpuID] = tree;
    eachExploredSol[gpuID] = sol;
    eachBest[gpuID] = best_l;

    deleteSinglePool(&pool_loc);
  }
  endTime = omp_get_wtime();
  double t2 = endTime - startTime;

  
  for (int i = 0; i < D; i++) {
    *exploredTree += eachExploredTree[i];
    *exploredSol += eachExploredSol[i];
  }
  
  // FIX THAT!
  best = (min reduce eachBest);
  
  printf("Workload per GPU: %f", (double)100*eachExploredTree/(exploredTree));

  printf("\nSearch on GPU completed\n");
  printf("Size of the explored tree: %llu\n", *exploredTree);
  printf("Number of explored solutions: %llu\n", *exploredSol);
  printf("Elapsed time: %f [s]\n", t2);

  /*
    Step 3: We complete the depth-first search on CPU.
  */

  startTime = omp_get_wtime();
  while (1) {
    int hasWork = 0;
    Node parent = popBack(&pool, &hasWork);
    if (!hasWork) break;

    decompose(jobs, lb, best, lbound1, lbound2, parent, exploredTree, exploredSol, &pool);
  }
  endTime = omp_get_wtime();
  double t3 = endTime - startTime;
  *elapsedTime = t1 + t2 + t3;
  printf("\nSearch on CPU completed\n");
  printf("Size of the explored tree: %llu\n", *exploredTree);
  printf("Number of explored solutions: %llu\n", *exploredSol);
  printf("Elapsed time: %f [s]\n", t3);

  printf("\nExploration terminated.\n");
  // printf("Cuda kernel calls: %d\n", count);

  deleteSinglePool(&pool);
  
  
  //clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  //*elapsedTime = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  printf("\nExploration terminated.\n");
  //printf("\n%d while loops execute.\n", count);

  // Freeing memory for structs 
  //deleteSinglePool(&pool);
  free_bound_data(lbound1);
  free_johnson_bd_data(lbound2);

  /* // Freeing memory for device */
  //cudaFree(parents_d);
  //cudaFree(bounds_d);
 

  /* //Freeing memory for host */
  //free(parents_h);
  //free(parents);
  //free(bounds);
}

int main(int argc, char* argv[])
{
  int inst, lb, ub, m, M, nbGPU;
  printf("I am before parse_parameters\n");
  parse_parameters(argc, argv, &inst, &lb, &ub, &m, &M, &nbGPU);
  printf("I am past parse_parameters\n");
  //exit(1);

  int jobs = taillard_get_nb_jobs(inst);
  int machines = taillard_get_nb_machines(inst);

  print_settings(inst, machines, jobs, ub, lb, nbGPU);

  int optimum = (ub == 1) ? taillard_get_best_ub(inst) : INT_MAX;
  unsigned long long int exploredTree = 0;
  unsigned long long int exploredSol = 0;

  double elapsedTime;

  pfsp_search(inst, lb, m, M, nbGPU, &optimum, &exploredTree, &exploredSol, &elapsedTime);

  print_results(optimum, exploredTree, exploredSol, elapsedTime);

  printf("We are done\n");

  return 0;
}
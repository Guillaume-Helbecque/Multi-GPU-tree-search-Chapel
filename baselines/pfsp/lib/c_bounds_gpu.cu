#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <string.h>

#include "c_bound_simple.h"
#include "c_bound_johnson.h"
#include "PFSP_node.h"

//Max size vectors for internal declarations in library
//#define MAX_MACHINES 40
//#define MAX_JOBS 40


//---------------One-machine bound functions-------------------

__device__ inline void
add_forward_gpu(const int job, const int * const p_times, const int nb_jobs, const int nb_machines, int * front)
{
  front[0] += p_times[job];
  for (int j = 1; j < nb_machines; j++) {
    front[j] = MAX(front[j - 1], front[j]) + p_times[j * nb_jobs + job];
  }
}

__device__ inline void
add_backward_gpu(const int job, const int * const p_times, const int nb_jobs, const int nb_machines, int * back)
{
  int j = nb_machines - 1;

  back[j] += p_times[j * nb_jobs + job];
  for (int j = nb_machines - 2; j >= 0; j--) {
    back[j] = MAX(back[j], back[j + 1]) + p_times[j * nb_jobs + job];
  }
}

__device__ void
schedule_front_gpu(const lb1_bound_data* lb1_data, const int * const permutation, const int limit1, int * front)
{
  const int N = lb1_data->nb_jobs;
  const int M = lb1_data->nb_machines;
  const int *const p_times = lb1_data->p_times;

  if (limit1 == -1) {
    for (int i = 0; i < M; i++)
      front[i] = lb1_data->min_heads[i];
    return;
  }
  for (int i = 0; i < M; i++){
    front[i] = 0;
  }
  for (int i = 0; i < limit1 + 1; i++) {
    add_forward_gpu(permutation[i], p_times, N, M, front);
  }
}

__device__ void
schedule_back_gpu(const lb1_bound_data* lb1_data, const int * const permutation, const int limit2, int * back)
{
  const int N = lb1_data->nb_jobs;
  const int M = lb1_data->nb_machines;
  const int *const p_times = lb1_data->p_times;

  if (limit2 == N) {
    for (int i = 0; i < M; i++)
      back[i] = lb1_data->min_tails[i];
    return;
  }

  for (int i = 0; i < M; i++) {
    back[i] = 0;
  }
  for (int k = N - 1; k >= limit2; k--) {
    add_backward_gpu(permutation[k], p_times, N, M, back);
  }
}

__device__ void
sum_unscheduled_gpu(const lb1_bound_data* lb1_data, const int * const permutation, const int limit1, const int limit2, int * remain)
{
  int nb_jobs = lb1_data->nb_jobs;
  int nb_machines = lb1_data->nb_machines;
  const int * const p_times = lb1_data->p_times;

  //memset(remain, 0, nb_machines*sizeof(int));
  
  for (int j = 0; j < nb_machines; j++) {
    remain[j] = 0;
  }
  
  for (int k = limit1 + 1; k < limit2; k++) {
    const int job = permutation[k];
    for (int j = 0; j < nb_machines; j++) {
      remain[j] += p_times[j * nb_jobs + job];
    }
  }
}

__device__ int
machine_bound_from_parts_gpu(const int * const front, const int * const back, const int * const remain, const int nb_machines)
{
  int tmp0 = front[0] + remain[0];
  int lb = tmp0 + back[0]; // LB on machine 0
  int tmp1;

  for (int i = 1; i < nb_machines; i++) {
    tmp1 = MAX(tmp0, front[i] + remain[i]);
    lb = MAX(lb, tmp1 + back[i]);
    tmp0 = tmp1;
  }

  return lb;
}

// adds job to partial schedule in front and computes lower bound on optimal cost
// NB1: schedule is no longer needed at this point
// NB2: front, remain and back need to be set before calling this
// NB3: also compute total idle time added to partial schedule (can be used a criterion for job ordering)
// nOps : m*(3 add+2 max)  ---> O(m)
__device__ int
add_front_and_bound_gpu(const lb1_bound_data* lb1_data, const int job, const int * const front, const int * const back, const int * const remain/*, int *delta_idle*/)
{
  int nb_jobs = lb1_data->nb_jobs;
  int nb_machines = lb1_data->nb_machines;
  int* p_times = lb1_data->p_times;

  int lb   = front[0] + remain[0] + back[0];
  int tmp0 = front[0] + p_times[job];
  int tmp1;

  int idle = 0;
  for (int i = 1; i < nb_machines; i++) {
    idle += MAX(0, tmp0 - front[i]);

    tmp1 = MAX(tmp0, front[i]);
    lb   = MAX(lb, tmp1 + remain[i] + back[i]);
    tmp0 = tmp1 + p_times[i * nb_jobs + job];
  }

  //can pass NULL
  // if (delta_idle) {
  //   delta_idle[job] = idle;
  // }

  return lb;
}

// ... same for back
__device__ int
add_back_and_bound_gpu(const lb1_bound_data* lb1_data, const int job, const int * const front, const int * const back, const int * const remain, int *delta_idle)
{
  int nb_jobs = lb1_data->nb_jobs;
  int nb_machines = lb1_data->nb_machines;
  int* p_times = lb1_data->p_times;

  int last_machine = nb_machines - 1;

  int lb   = front[last_machine] + remain[last_machine] + back[last_machine];
  int tmp0 = back[last_machine] + p_times[last_machine*nb_jobs + job];
  int tmp1;

  int idle = 0;
  for (int i = last_machine-1; i >= 0; i--) {
    idle += MAX(0, tmp0 - back[i]);

    tmp1 = MAX(tmp0, back[i]);
    lb = MAX(lb, tmp1 + remain[i] + front[i]);
    tmp0 = tmp1 + p_times[i*nb_jobs + job];
  }

  //can pass NULL
  if (delta_idle) {
    delta_idle[job] = idle;
  }

  return lb;
}

__device__ void
lb1_bound_gpu(const lb1_bound_data* lb1_data, const int * const permutation, const int limit1, const int limit2, int *bounds)
{
  int nb_machines = lb1_data->nb_machines;

  int front[MAX_MACHINES];
  int back[MAX_MACHINES];
  int remain[MAX_MACHINES];
  
  schedule_front_gpu(lb1_data, permutation, limit1, front);
  schedule_back_gpu(lb1_data, permutation, limit2, back);

  sum_unscheduled_gpu(lb1_data, permutation, limit1, limit2, remain);

  // Same as in function eval_solution_gpu
  *bounds = machine_bound_from_parts_gpu(front, back, remain, nb_machines);

  return;
}

__device__ void lb1_children_bounds_gpu(const lb1_bound_data* lb1_data, const int *const permutation, const int limit1, const int limit2, int *const lb_begin/*, int *const lb_end, int *const prio_begin, int *const prio_end, const int direction*/)
{
  int N = lb1_data->nb_jobs;
  
  int front[MAX_MACHINES];
  int back[MAX_MACHINES];
  int remain[MAX_MACHINES];

  schedule_front_gpu(lb1_data, permutation, limit1, front);
  schedule_back_gpu(lb1_data, permutation, limit2, back);
  sum_unscheduled_gpu(lb1_data, permutation, limit1, limit2, remain);

  // switch (direction)  {
  //   case -1: //begin
  //   {
  //memset(lb_begin, 0, N*sizeof(int));
  // if (prio_begin) memset(prio_begin, 0, N*sizeof(int));
  for(int i = 0; i < N; i++)
    lb_begin[i]=0;

  
  for (int i = limit1+1; i < limit2; i++) {
    int job = permutation[i];
    lb_begin[job] = add_front_and_bound_gpu(lb1_data, job, front, back, remain/*, prio_begin*/);
  }
  //     break;
  //   }
  //   case 0: //begin-end
  //   {
  //     memset(lb_begin, 0, N*sizeof(int));
  //     memset(lb_end, 0, N*sizeof(int));
  //     if (prio_begin) memset(prio_begin, 0, N*sizeof(int));
  //     if (prio_end) memset(prio_end, 0, N*sizeof(int));
  //
  //     for (int i = limit1+1; i < limit2; i++) {
  //       int job = permutation[i];
  //       lb_begin[job] = add_front_and_bound(data, job, front, back, remain, prio_begin);
  //       lb_end[job] = add_back_and_bound(data, job, front, back, remain, prio_end);
  //     }
  //     break;
  //   }
  //   case 1: //end
  //   {
  //     memset(lb_end, 0, N*sizeof(int));
  //     if (prio_end) memset(prio_end, 0, N*sizeof(int));
  //
  //     for (int i = limit1+1; i < limit2; i++) {
  //       int job = permutation[i];
  //       lb_end[job] = add_back_and_bound(data, job, front, back, remain, prio_end);
  //     }
  //     break;
  //   }
  // }
}

//------------------Two-machine bound functions(johnson)---------------------------

__device__ int lb_makespan_gpu(int* lb1_p_times, const lb2_bound_data* lb2_data, const int* const flag,int* front, int* back, const int minCmax)
{
  int lb = 0;
  int nb_jobs = lb2_data->nb_jobs;
  
  //for all machine-pairs : O(m^2) m*(m-1)/2
  for (int l = 0; l < lb2_data->nb_machine_pairs; l++) {
    int i = lb2_data->machine_pair_order[l];

    int ma0 = lb2_data->machine_pairs_1[i];
    int ma1 = lb2_data->machine_pairs_2[i];

    int tmp0 = front[ma0];
    int tmp1 = front[ma1];

    // compute_cmax_johnson
    for (int j = 0; j < nb_jobs; j++) {
      int job = lb2_data->johnson_schedules[i*nb_jobs + j];
      // j-loop is on unscheduled jobs... (==0 if jobCour is unscheduled)
      if (flag[job] == 0) {
	int ptm0 = lb1_p_times[ma0*nb_jobs + job];
	int ptm1 = lb1_p_times[ma1*nb_jobs + job];
	int lag = lb2_data->lags[i*nb_jobs + job];
	// add job on ma0 and ma1
	tmp0 += ptm0;
	tmp1 = MAX(tmp1,tmp0 + lag);
	tmp1 += ptm1;
      }
    }
    // end compute_cmax_johnson
    
    tmp1 = MAX(tmp1 + back[ma1], tmp0 + back[ma0]);

    lb = MAX(lb, tmp1);
    
    if (lb > minCmax) {
      break;
    }
  }

  return lb;
}

__device__ void lb2_bound_gpu(const lb1_bound_data* lb1_data, const lb2_bound_data* lb2_data, const int* const permutation, const int limit1, const int limit2,const int best_cmax,int *bounds)
{
  const int N = lb1_data->nb_jobs;

  int front[MAX_MACHINES];
  int back[MAX_MACHINES];

  schedule_front_gpu(lb1_data, permutation, limit1, front);
  schedule_back_gpu(lb1_data, permutation, limit2, back);

  int flags[MAX_JOBS];
   
  // Set flags
  for (int i = 0; i < N; i++)
    flags[i] = 0;
  for (int j = 0; j <= limit1; j++)
    flags[permutation[j]] = 1;
  for (int j = limit2; j < N; j++)
    flags[permutation[j]] = 1;
  // end flags

  *bounds = lb_makespan_gpu(lb1_data->p_times, lb2_data, flags, front, back, best_cmax);
  return;
}


//-----------------UNUSED FUNCTIONS---------------

//-----------------FOR SIMPLE BOUNDING----------------

// //----------------------evaluate (partial) schedules---------------------
// __device__ int eval_solution_gpu(const lb1_bound_data lb1_data, const int* const permutation)
// {
//   const int N = lb1_data.nb_jobs;
//   const int M = lb1_data.nb_machines;

//   //int tmp[N];
//   //int *tmp = (int*)malloc(N * sizeof(int)); // Dynamically allocate memory for tmp
//   int *tmp;
//   cudaMalloc((void**)&tmp, N * sizeof(int));

//   int result;
  
//   // Check if memory allocation succeeded
//   if(tmp == NULL) {
//     // Handle memory allocation failure
//     return -1; // Return an error code indicating failure
//   }
  
//   for(int i = 0; i < N; i++) {
//     tmp[i] = 0;
//   }
//   for (int i = 0; i < N; i++) {
//     add_forward_gpu(permutation[i], lb1_data.p_times, N, M, tmp);
//   }

//   // In order to free tmp, we have to put the value of return in an auxiliary variable called result
//   result = tmp[M-1];
//   //cudaFree(tmp);
//   return result;
// }


//-----------------FOR JOHSON BOUNDING-----------------

// __device__ inline int compute_cmax_johnson_gpu(const int* const lb1_p_times, const lb2_bound_data lb2_data, const int* const flag, int *tmp0, int *tmp1, int ma0, int ma1, int ind)
// {
//   int nb_jobs = lb2_data.nb_jobs;
    
//   for (int j = 0; j < nb_jobs; j++) {
//     int job = lb2_data.johnson_schedules[ind*nb_jobs + j];
//     // j-loop is on unscheduled jobs... (==0 if jobCour is unscheduled)
//     if (flag[job] == 0) {
//       int ptm0 = lb1_p_times[ma0*nb_jobs + job];
//       int ptm1 = lb1_p_times[ma1*nb_jobs + job];
//       int lag = lb2_data.lags[ind*nb_jobs + job];
//       // add job on ma0 and ma1
//       *tmp0 += ptm0;
//       *tmp1 = MAX(*tmp1,*tmp0 + lag);
//       *tmp1 += ptm1;
//     }
//   }

//   return *tmp1;
// }

// __device__ void set_flags_gpu(const int *const permutation, const int limit1, const int limit2, const int N, int* flags)
// {
//   for (int i = 0; i < N; i++)
//     flags[i] = 0;
//   for (int j = 0; j <= limit1; j++)
//     flags[permutation[j]] = 1;
//   for (int j = limit2; j < N; j++)
//     flags[permutation[j]] = 1;
// }

// //allows variable nb of machine pairs and get machine pair the realized best lb
// __device__ int lb_makespan_learn_gpu(const int* const lb1_p_times, const lb2_bound_data lb2_data, const int* const flag, const int* const front, const int* const back, const int minCmax, const int nb_pairs, int *best_index)
// {
//   int lb = 0;

//   for (int l = 0; l < nb_pairs; l++) {
//     int i = lb2_data.machine_pair_order[l];

//     int ma0 = lb2_data.machine_pairs_1[i];
//     int ma1 = lb2_data.machine_pairs_2[i];

//     int tmp0 = front[ma0];
//     int tmp1 = front[ma1];

//     compute_cmax_johnson_gpu(lb1_p_times, lb2_data, flag, &tmp0, &tmp1, ma0, ma1, i);

//     tmp1 = MAX(tmp1 + back[ma1], tmp0 + back[ma0]);

//     if (tmp1 > lb) {
//       *best_index = i;
//       lb = tmp1;
//     }
//     // lb=MAX(lb,tmp1);

//     if (lb > minCmax) {
//       break;
//     }
//   }

//   return lb;
// }

// __device__ void lb2_children_bounds_gpu(const lb1_bound_data lb1_data, const lb2_bound_data lb2_data, const int* const permutation, const int limit1, const int limit2, int* const lb_begin, int* const lb_end, const int best_cmax, const int direction)
// {
//   const int N = lb1_data.nb_jobs;
  
//   //int *tmp_perm = (int*)malloc(N * sizeof(int)); // Dynamically allocate memory for tmp_perm

//   int tmp_perm[MAX_JOBS];
  
//   // // Check if memory allocation succeeded
//   // if(tmp_perm == NULL) {
//   //   // Handle memory allocation failure
//   //   return; // Return an error code indicating failure
//   // }

//   memcpy(tmp_perm, permutation, N*sizeof(int));

//   switch (direction) {
//     case -1:
//      {
//       for (int i = limit1 + 1; i < limit2; i++) {
//         int job = tmp_perm[i];

//         swap(&tmp_perm[i], &tmp_perm[limit1 + 1]);
//         lb_begin[job] = lb2_bound_gpu(lb1_data, lb2_data, tmp_perm, limit1+1, limit2, best_cmax);
//         swap(&tmp_perm[i], &tmp_perm[limit1 + 1]);
//       }
//       break;
//     }
//     case 0:
//     {
//       for (int i = limit1 + 1; i < limit2; i++) {
//         int job = tmp_perm[i];

//         swap(&tmp_perm[i], &tmp_perm[limit1 + 1]);
//         lb_begin[job] = lb2_bound_gpu(lb1_data, lb2_data, tmp_perm, limit1+1, limit2, best_cmax);
//         swap(&tmp_perm[i], &tmp_perm[limit1 + 1]);

//         swap(&tmp_perm[i], &tmp_perm[limit2 - 1]);
//         lb_end[job] = lb2_bound_gpu(lb1_data, lb2_data, tmp_perm, limit1, limit2-1, best_cmax);
//         swap(&tmp_perm[i], &tmp_perm[limit2 - 1]);
//       }
//       break;
//     }
//     case 1:
//     {
//       for (int i = limit1 + 1; i < limit2; i++) {
//         int job = tmp_perm[i];

//         swap(&tmp_perm[i], &tmp_perm[limit2 - 1]);
//         lb_end[job] = lb2_bound_gpu(lb1_data, lb2_data, tmp_perm, limit1, limit2-1, best_cmax);
//         swap(&tmp_perm[i], &tmp_perm[limit2 - 1]);
//       }
//       break;
//     }
//   }
// }

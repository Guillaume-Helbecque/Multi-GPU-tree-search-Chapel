/*
  Multi-GPU B&B to solve Taillard instances of the PFSP in Chapel.
*/

use Time;
use Random;
use GpuDiagnostics;

config const BLOCK_SIZE = 512;

use util;
use Pool_par;

use PFSP_node;
use Bound_johnson;
use Bound_simple;
use Taillard;

/*******************************************************************************
Implementation of the multi-GPU PFSP search.
*******************************************************************************/

config const m = 25;
config const M = 50000;
config const D = 1;

config const inst: int = 14; // instance
config const lb: int = 1; // lower bound function
config const ub: int = 1; // initial upper bound
/*
  NOTE: Only forward branching is considered because other strategies increase a
  lot the implementation complexity and do not add much contribution.
*/

const jobs = taillard_get_nb_jobs(inst);
const machines = taillard_get_nb_machines(inst);

const initUB = if (ub == 1) then taillard_get_best_ub(inst) else max(int);

proc check_parameters()
{
  if ((m <= 0) || (M <= 0) || (D <= 0)) then
    halt("Error: m, M, and D must be positive integers.\n");

  if (inst < 1 || inst > 120) then
    halt("Error: unsupported Taillard's instance");

  if (lb < 0 || lb > 2) then
    halt("Error: unsupported lower bound function");

  if (ub != 0 && ub != 1) then
    halt("Error: unsupported upper bound initialization");
}

proc print_settings(): void
{
  writeln("\n=================================================");
  writeln("Multi-GPU Chapel (", D, " GPUs)\n");
  writeln("Resolution of PFSP Taillard's instance: ta", inst, " (m = ", machines, ", n = ", jobs, ")");
  if (ub == 0) then writeln("Initial upper bound: inf");
  else /* if (ub == 1) */ writeln("Initial upper bound: opt");
  if (lb == 0) then writeln("Lower bound function: lb1_d");
  else if (lb == 1) then writeln("Lower bound function: lb1");
  else /* if (lb == 2) */ writeln("Lower bound function: lb2");
  writeln("Branching rule: fwd");
  writeln("=================================================");
}

proc print_results(const optimum: int, const exploredTree: uint, const exploredSol: uint,
  const timer: real)
{
  writeln("\n=================================================");
  writeln("Size of the explored tree: ", exploredTree);
  writeln("Number of explored solutions: ", exploredSol);
  const is_better = if (optimum < initUB) then " (improved)"
                                          else " (not improved)";
  writeln("Optimal makespan: ", optimum, is_better);
  writeln("Elapsed time: ", timer, " [s]");
  writeln("=================================================\n");
}

// Evaluate and generate children nodes on CPU.
proc decompose_lb1(const lb1_data, const parent: Node, ref tree_loc: uint, ref num_sol: uint,
  ref best: int, ref pool: SinglePool_par(Node))
{
  for i in parent.limit1+1..(jobs-1) {
    var child = new Node();
    child.depth = parent.depth + 1;
    child.limit1 = parent.limit1 + 1;
    child.prmu = parent.prmu;
    child.prmu[parent.depth] <=> child.prmu[i];

    var lowerbound = lb1_bound(lb1_data, child.prmu, child.limit1, jobs);

    if (child.depth == jobs) { // if child leaf
      num_sol += 1;

      if (lowerbound < best) { // if child feasible
        best = lowerbound;
      }
    } else { // if not leaf
      if (lowerbound < best) { // if child feasible
        pool.pushBackFree(child);
        tree_loc += 1;
      }
    }
  }
}

proc decompose_lb1_d(const lb1_data, const parent: Node, ref tree_loc: uint, ref num_sol: uint,
  ref best: int, ref pool: SinglePool_par(Node))
{
  var lb_begin: MAX_JOBS*int(32);

  lb1_children_bounds(lb1_data, parent.prmu, parent.limit1, jobs, lb_begin);

  for i in parent.limit1+1..(jobs-1) {
    const job = parent.prmu[i];
    const lowerbound = lb_begin[job];

    if (parent.depth + 1 == jobs) { // if child leaf
      num_sol += 1;

      if (lowerbound < best) { // if child feasible
        best = lowerbound;
      }
    } else { // if not leaf
      if (lowerbound < best) { // if child feasible
        var child = new Node();
        child.depth = parent.depth + 1;
        child.limit1 = parent.limit1 + 1;
        child.prmu = parent.prmu;
        child.prmu[parent.depth] <=> child.prmu[i];

        pool.pushBackFree(child);
        tree_loc += 1;
      }
    }
  }
}

proc decompose_lb2(const lb1_data, const lb2_data, const parent: Node, ref tree_loc: uint, ref num_sol: uint,
  ref best: int, ref pool: SinglePool_par(Node))
{
  for i in parent.limit1+1..(jobs-1) {
    var child = new Node();
    child.depth = parent.depth + 1;
    child.limit1 = parent.limit1 + 1;
    child.prmu = parent.prmu;
    child.prmu[parent.depth] <=> child.prmu[i];

    var lowerbound = lb2_bound(lb1_data, lb2_data, child.prmu, child.limit1, jobs, best);

    if (child.depth == jobs) { // if child leaf
      num_sol += 1;

      if (lowerbound < best) { // if child feasible
        best = lowerbound;
      }
    } else { // if not leaf
      if (lowerbound < best) { // if child feasible
        pool.pushBackFree(child);
        tree_loc += 1;
      }
    }
  }
}

// Evaluate and generate children nodes on CPU.
proc decompose(const lb1_data, const lb2_data, const parent: Node, ref tree_loc: uint, ref num_sol: uint,
  ref best: int, ref pool: SinglePool_par(Node))
{
  select lb {
    when 0 {
      decompose_lb1_d(lb1_data, parent, tree_loc, num_sol, best, pool);
    }
    when 1 {
      decompose_lb1(lb1_data, parent, tree_loc, num_sol, best, pool);
    }
    otherwise { // 2
      decompose_lb2(lb1_data, lb2_data, parent, tree_loc, num_sol, best, pool);
    }
  }
}

// Evaluate a bulk of parent nodes on GPU using lb1.
proc evaluate_gpu_lb1(const parents_d: [] Node, const size, const lbound1_d, ref bounds_d)
{
  @assertOnGpu
  foreach threadId in 0..#size {
    const parentId = threadId / jobs;
    const k = threadId % jobs;
    var parent = parents_d[parentId];
    const depth = parent.depth;
    var prmu = parent.prmu;

    if (k >= parent.limit1+1) {
      prmu[depth] <=> prmu[k];
      bounds_d[threadId] = lb1_bound(lbound1_d, prmu, parent.limit1+1, jobs);
      prmu[depth] <=> prmu[k];
    }
  }
}

/*
  NOTE: This lower bound evaluates all the children of a given parent at the same time.
  Therefore, the GPU loop is on the parent nodes and not on the children ones, in contrast
  to the other lower bounds.
*/
// Evaluate a bulk of parent nodes on GPU using lb1_d.
proc evaluate_gpu_lb1_d(const parents_d: [] Node, const size, const best, const lbound1_d, ref bounds_d)
{
  @assertOnGpu
  foreach parentId in 0..#(size/jobs) {
    var parent = parents_d[parentId];
    const depth = parent.depth;
    var prmu = parent.prmu;

    var lb_begin: MAX_JOBS*int(32);

    lb1_children_bounds(lbound1_d, parent.prmu, parent.limit1, jobs, lb_begin);

    for k in 0..#jobs {
      if (k >= parent.limit1+1) {
        const job = parent.prmu[k];
        bounds_d[parentId*jobs+k] = lb_begin[job];
      }
    }
  }
}

// Evaluate a bulk of parent nodes on GPU using lb2.
proc evaluate_gpu_lb2(const parents_d: [] Node, const size, const best, const lbound1_d, const lbound2_d, ref bounds_d)
{
  @assertOnGpu
  foreach threadId in 0..#size {
    const parentId = threadId / jobs;
    const k = threadId % jobs;
    var parent = parents_d[parentId];
    const depth = parent.depth;
    var prmu = parent.prmu;

    if (k >= parent.limit1+1) {
      prmu[depth] <=> prmu[k];
      bounds_d[threadId] = lb2_bound(lbound1_d, lbound2_d, prmu, parent.limit1+1, jobs, best);
      prmu[depth] <=> prmu[k];
    }
  }
}

// Evaluate a bulk of parent nodes on GPU.
proc evaluate_gpu(const parents_d: [] Node, const size, const best, const lbound1_d, const lbound2_d, ref bounds_d)
{
  select lb {
    when 0 {
      evaluate_gpu_lb1_d(parents_d, size, best, lbound1_d!.lb1_bound, bounds_d);
    }
    when 1 {
      evaluate_gpu_lb1(parents_d, size, lbound1_d!.lb1_bound, bounds_d);
    }
    otherwise { // 2
      evaluate_gpu_lb2(parents_d, size, best, lbound1_d!.lb1_bound, lbound2_d!.lb2_bound, bounds_d);
    }
  }
}

// Generate children nodes (evaluated by GPU) on CPU.
proc generate_children(const ref parents: [] Node, const size: int, const ref bounds: [] int(32),
  ref exploredTree: uint, ref exploredSol: uint, ref best: int, ref pool: SinglePool_par(Node))
{
  for i in 0..#size {
    const parent = parents[i];
    const depth = parent.depth;

    for j in parent.limit1+1..(jobs-1) {
      const lowerbound = bounds[j + i * jobs];

      if (depth + 1 == jobs) { // if child leaf
        exploredSol += 1;

        if (lowerbound < best) { // if child feasible
          best = lowerbound;
        }
      } else { // if not leaf
        if (lowerbound < best) { // if child feasible
          var child = new Node();
          child.depth = parent.depth + 1;
          child.limit1 = parent.limit1 + 1;
          child.prmu = parent.prmu;
          child.prmu[parent.depth] <=> child.prmu[j];

          pool.pushBack(child);
          exploredTree += 1;
        }
      }
    }
  }
}

class WrapperClassArrayParents {
  forwarding var arr: [0..#M] Node = noinit;
}
type WrapperArrayParents = owned WrapperClassArrayParents?;

class WrapperClassArrayBounds {
  forwarding var arr: [0..#(M*jobs)] int(32) = noinit;
}
type WrapperArrayBounds = owned WrapperClassArrayBounds?;

// Multi-GPU PFSP search.
proc pfsp_search(ref optimum: int, ref exploredTree: uint, ref exploredSol: uint, ref elapsedTime: real)
{
  var best: int = initUB;

  var root = new Node(jobs);

  var pool = new SinglePool_par(Node);
  pool.pushBackFree(root);

  var allTasksIdleFlag: atomic bool = false;
  var eachTaskState: [0..#D] atomic bool = BUSY; // one task per GPU

  var timer: stopwatch;

  /*
    Step 1: We perform a partial breadth-first search on CPU in order to create
    a sufficiently large amount of work for GPU computation.
  */
  timer.start();

  var lbound1 = new lb1_bound_data(jobs, machines);
  taillard_get_processing_times(lbound1.p_times, inst);
  fill_min_heads_tails(lbound1);

  var lbound2 = new lb2_bound_data(jobs, machines);
  fill_machine_pairs(lbound2/*, LB2_FULL*/);
  fill_lags(lbound1.p_times, lbound2);
  fill_johnson_schedules(lbound1.p_times, lbound2);

  while (pool.size < D*m) {
    var hasWork = 0;
    var parent = pool.popFrontFree(hasWork);
    if !hasWork then break;

    decompose(lbound1, lbound2, parent, exploredTree, exploredSol, best, pool);
  }
  timer.stop();
  const res1 = (timer.elapsed(), exploredTree, exploredSol);
  writeln("\nInitial search on CPU completed");
  writeln("Size of the explored tree: ", res1[1]);
  writeln("Number of explored solutions: ", res1[2]);
  writeln("Elapsed time: ", res1[0], " [s]\n");

  /*
    Step 2: We continue the search on GPU in a depth-first manner, until there
    is not enough work.
  */
  timer.start();
  var eachExploredTree, eachExploredSol: [0..#D] uint = noinit;
  var eachBest: [0..#D] int = noinit;

  const poolSize = pool.size;
  const c = poolSize / D;
  const l = poolSize - (D-1)*c;
  const f = pool.front;

  pool.front = 0;
  pool.size = 0;

  var multiPool: [0..#D] SinglePool_par(Node);

  coforall gpuID in 0..#D with (ref pool, ref eachExploredTree, ref eachExploredSol,
    ref eachBest, ref eachTaskState, ref multiPool) {

    const device = here.gpus[gpuID];

    var nSteal, nSSteal: int;

    var tree, sol: uint;
    ref pool_loc = multiPool[gpuID];
    var best_l = best;
    var taskState: bool = BUSY;

    // each task gets its chunk
    pool_loc.elements[0..#c] = pool.elements[gpuID+f.. by D #c];
    pool_loc.size += c;
    if (gpuID == D-1) {
      pool_loc.elements[c..#(l-c)] = pool.elements[(D*c)+f..#(l-c)];
      pool_loc.size += l-c;
    }

    var parents: [0..#M] Node = noinit;
    var bounds: [0..#(M*jobs)] int(32) = noinit;

    var lbound1_d: WrapperLB1;
    var lbound2_d: WrapperLB2;
    var parents_d: WrapperArrayParents;
    var bounds_d: WrapperArrayBounds;

    on device {
      lbound1_d = new WrapperLB1(jobs, machines);
      lbound1_d!.lb1_bound.p_times   = lbound1.p_times;
      lbound1_d!.lb1_bound.min_heads = lbound1.min_heads;
      lbound1_d!.lb1_bound.min_tails = lbound1.min_tails;

      lbound2_d = new WrapperLB2(jobs, machines);
      lbound2_d!.lb2_bound.johnson_schedules  = lbound2.johnson_schedules;
      lbound2_d!.lb2_bound.lags               = lbound2.lags;
      lbound2_d!.lb2_bound.machine_pairs      = lbound2.machine_pairs;
      lbound2_d!.lb2_bound.machine_pair_order = lbound2.machine_pair_order;

      parents_d = new WrapperArrayParents();
      bounds_d = new WrapperArrayBounds();
    }

    while true {
      /*
        Each task gets its parents nodes from the pool.
      */
      var poolSize = pool_loc.popBackBulk(m, M, parents);

      if (poolSize > 0) {
        if (taskState == IDLE) {
          taskState = BUSY;
          eachTaskState[gpuID].write(BUSY);
        }

        /* poolSize = min(poolSize, M);
        var hasWork = 0;
        var parents: [0..#poolSize] Node = noinit;
        pool_loc.popBackBulk(poolSize, parents, hasWork);
        if (hasWork == 0) {
          writeln("DEADCODE in get parents");
          break;
        } */

        /*
          TODO: Optimize 'numBounds' based on the fact that the maximum number of
          generated children for a parent is 'parent.limit2 - parent.limit1 + 1' or
          something like that.
        */
        const numBounds = jobs * poolSize;

        on device {
          parents_d!.arr = parents; // host-to-device
          evaluate_gpu(parents_d!.arr, numBounds, best_l, lbound1_d, lbound2_d, bounds_d!.arr);
          bounds = bounds_d!.arr; // device-to-host
        }

        /*
          Each task generates and inserts its children nodes to the pool.
        */
        generate_children(parents, poolSize, bounds, tree, sol, best_l, pool_loc);
      }
      else {
        // work stealing
        var tries = 0;
        var steal = false;
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
                  var (hasWork, p) = victim.popFrontBulkFree(m, M);
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
            writeln("task ", gpuID, " exits normally");
            break;
          }
          continue;
        } else {
          continue;
        }
      }
    }

    const poolLocSize = pool_loc.size;
    for p in 0..#poolLocSize {
      var hasWork = 0;
      pool.pushBack(pool_loc.popBack(hasWork));
      if !hasWork then break;
    }

    eachExploredTree[gpuID] = tree;
    eachExploredSol[gpuID] = sol;
    eachBest[gpuID] = best_l;

    writeln("work stealing tries on tasks ", gpuID, " : ", nSteal, " and successes : ", nSSteal);
  }
  timer.stop();

  exploredTree += (+ reduce eachExploredTree);
  exploredSol += (+ reduce eachExploredSol);
  best = (min reduce eachBest);

  writeln("workload per GPU: ", 100.0*eachExploredTree/(exploredTree-res1[1]):real);

  const res2 = (timer.elapsed(), exploredTree, exploredSol) - res1;
  writeln("Search on GPU completed");
  writeln("Size of the explored tree: ", res2[1]);
  writeln("Number of explored solutions: ", res2[2]);
  writeln("Elapsed time: ", res2[0], " [s]\n");

  /*
    Step 3: We complete the depth-first search on CPU.
  */
  timer.start();
  while true {
    var hasWork = 0;
    var parent = pool.popBackFree(hasWork);
    if !hasWork then break;

    decompose(lbound1, lbound2, parent, exploredTree, exploredSol, best, pool);
  }
  timer.stop();
  elapsedTime = timer.elapsed();
  const res3 = (elapsedTime, exploredTree, exploredSol) - res1 - res2;
  writeln("Search on CPU completed");
  writeln("Size of the explored tree: ", res3[1]);
  writeln("Number of explored solutions: ", res3[2]);
  writeln("Elapsed time: ", res3[0], " [s]");

  optimum = best;

  writeln("\nExploration terminated.");
}

proc main()
{
  check_parameters();
  print_settings();

  var optimum: int;
  var exploredTree: uint = 0;
  var exploredSol: uint = 0;

  var elapsedTime: real;

  startGpuDiagnostics();

  pfsp_search(optimum, exploredTree, exploredSol, elapsedTime);

  stopGpuDiagnostics();

  print_results(optimum, exploredTree, exploredSol, elapsedTime);

  writeln("GPU diagnostics:");
  writeln("   kernel_launch: ", getGpuDiagnostics().kernel_launch);
  writeln("   host_to_device: ", getGpuDiagnostics().host_to_device);
  writeln("   device_to_host: ", getGpuDiagnostics().device_to_host);
  writeln("   device_to_device: ", getGpuDiagnostics().device_to_device);

  return 0;
}

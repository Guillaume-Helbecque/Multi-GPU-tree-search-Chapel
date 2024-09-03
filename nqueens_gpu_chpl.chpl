/*
  Single-GPU backtracking to solve instances of the N-Queens problem in Chapel.
*/

use Time;

use Pool;
use GpuDiagnostics;

use NQueens_node;

config const BLOCK_SIZE = 512;

/*******************************************************************************
Implementation of the single-GPU N-Queens search.
*******************************************************************************/

config const N = 14;
config const g = 1;
config const m = 25;
config const M = 50000;

proc check_parameters()
{
  if ((N <= 0) || (g <= 0) || (m <= 0) || (M <= 0)) {
    halt("All parameters must be positive integers.\n");
  }
}

proc print_settings()
{
  writeln("\n=================================================");
  writeln("Single-GPU Chapel\n");
  writeln("Resolution of the ", N, "-Queens instance");
  writeln("  with ", g, " safety check(s) per evaluation");
  writeln("=================================================");
}

proc print_results(const exploredTree: uint, const exploredSol: uint, const timer: real)
{
  writeln("\n=================================================");
  writeln("Size of the explored tree: ", exploredTree);
  writeln("Number of explored solutions: ", exploredSol);
  writeln("Elapsed time: ", timer, " [s]");
  writeln("=================================================\n");
}

// Check queen's safety.
proc isSafe(const board, const queen_num, const row_pos): uint(8)
{
  var isSafe: uint(8) = 1;

  for i in 0..#queen_num {
    const other_row_pos = board[i];

    for _g in 0..#g {
      if (other_row_pos == row_pos - (queen_num - i) ||
          other_row_pos == row_pos + (queen_num - i)) {
        isSafe = 0;
      }
    }
  }

  return isSafe;
}

// Evaluate and generate children nodes on CPU.
proc decompose(const parent: Node, ref tree_loc: uint, ref num_sol: uint, ref pool: SinglePool(Node))
{
  const depth = parent.depth;

  if (depth == N) {
    num_sol += 1;
  }
  for j in depth..(N-1) {
    if isSafe(parent.board, depth, parent.board[j]) {
      var child = new Node();
      child.depth = parent.depth;
      child.board = parent.board;
      child.board[depth] <=> child.board[j];
      child.depth += 1;
      pool.pushBack(child);
      tree_loc += 1;
    }
  }
}

// Evaluate a bulk of parent nodes on GPU.
proc evaluate_gpu(const parents_d: [] Node, const size)
{
  var labels: [0..#size] uint(8) = noinit;

  @assertOnGpu
  foreach threadId in 0..#size {
    const parentId = threadId / N;
    const k = threadId % N;
    const parent = parents_d[parentId];
    const depth = parent.depth;
    const queen_num = parent.board[k];

    var isSafe: uint(8);
    // If child 'k' is not scheduled, we evaluate its safety 'G' times, otherwise 0.
    if (k >= depth) {
      isSafe = 1;
      /* const G_notScheduled = g * (k >= depth); */
      for i in 0..#depth {
        const pbi = parent.board[i];

        for _g in 0..#g {//G_notScheduled {
          isSafe *= (pbi != queen_num - (depth - i) &&
                     pbi != queen_num + (depth - i));
        }
      }
      labels[threadId] = isSafe;
    }
  }

  return labels;
}

// Generate children nodes (evaluated on GPU) on CPU.
proc generate_children(const ref parents: [] Node, const size: int, const ref labels: [] uint(8),
  ref exploredTree: uint, ref exploredSol: uint, ref pool: SinglePool(Node))
{
  for i in 0..#size  {
    const parent = parents[i];
    const depth = parent.depth;

    if (depth == N) {
      exploredSol += 1;
    }
    for j in depth..(N-1) {
      if (labels[j + i * N] == 1) {
        var child = new Node();
        child.depth = depth + 1;
        child.board = parent.board;
        child.board[depth] <=> child.board[j];
        pool.pushBack(child);
        exploredTree += 1;
      }
    }
  }
}

// Single-GPU N-Queens search.
proc nqueens_search(ref exploredTree: uint, ref exploredSol: uint, ref elapsedTime: real)
{
  var root = new Node(N);

  var pool = new SinglePool(Node);
  pool.pushBack(root);

  var timer: stopwatch;
  timer.start();

  while true {
    var hasWork = 0;
    var parent = pool.popBack(hasWork);
    if !hasWork then break;

    decompose(parent, exploredTree, exploredSol, pool);

    var poolSize = min(pool.size, M);

    // If 'poolSize' is sufficiently large, we offload the pool on GPU.
    if (poolSize >= m) {

      var parents: [0..#poolSize] Node = noinit;
      for i in 0..#poolSize {
        var hasWork = 0;
        parents[i] = pool.popBack(hasWork);
        if !hasWork then break;
      }

      const numLabels = N * poolSize;
      var labels: [0..#numLabels] uint(8) = noinit;

      on here.gpus[0] {
        const parents_d = parents; // host-to-device
        labels = evaluate_gpu(parents_d, numLabels);
      }

      /*
        Each task generates and inserts its children nodes to the pool.
      */
      generate_children(parents, poolSize, labels, exploredTree, exploredSol, pool);
    }
  }
  timer.stop();
  elapsedTime = timer.elapsed();

  writeln("\nExploration terminated.");
}

proc main()
{
  check_parameters();
  print_settings();

  var exploredTree: uint = 0;
  var exploredSol: uint = 0;

  var elapsedTime: real;

  startGpuDiagnostics();

  nqueens_search(exploredTree, exploredSol, elapsedTime);

  stopGpuDiagnostics();

  print_results(exploredTree, exploredSol, elapsedTime);

  writeln("GPU diagnostics:");
  writeln("   kernel_launch: ", getGpuDiagnostics().kernel_launch);
  writeln("   host_to_device: ", getGpuDiagnostics().host_to_device);
  writeln("   device_to_host: ", getGpuDiagnostics().device_to_host);
  writeln("   device_to_device: ", getGpuDiagnostics().device_to_device);

  return 0;
}

module Bound_johnson
{
  use Sort;
  use Bound_simple;

  enum lb2_variant { LB2_FULL, LB2_NABESHIMA, LB2_LAGEWEG, LB2_LEARN }

  param NUM_MACHINES = 20;
  param NUM_JOBS = 20;

  record lb2_bound_data
  {
    // constants
    var nb_jobs: int(32);
    var nb_machines: int(32);
    var nb_machine_pairs: int(32);
    // domains
    var ld: domain(1);
    var ad: domain(1);
    // data arrays
    var johnson_schedules: [ld] int(32);
    var lags: [ld] int(32);
    var machine_pairs: [0..1] [ad] int(32);
    var machine_pair_order: [ad] int(32);

    proc init() {}

    proc init(const jobs: int(32), const machines: int(32)/*, enum lb2_variant lb2_type*/)
    {
      this.nb_jobs = jobs;
      this.nb_machines = machines;

      var lb2_type = lb2_variant.LB2_FULL;

      //depends on nb of machine pairs
      if (lb2_type == lb2_variant.LB2_FULL) then
        this.nb_machine_pairs = (this.nb_machines*(this.nb_machines-1))/2;
      if (lb2_type == lb2_variant.LB2_NABESHIMA) then
        this.nb_machine_pairs = this.nb_machines-1;
      if (lb2_type == lb2_variant.LB2_LAGEWEG) then
        this.nb_machine_pairs = this.nb_machines-1;
      if (lb2_type == lb2_variant.LB2_LEARN) then
        this.nb_machine_pairs = (this.nb_machines*(this.nb_machines-1))/2;

      this.ld = {0..#this.nb_machine_pairs*this.nb_jobs};
      this.ad = {0..#this.nb_machine_pairs};
    }
  }

  /* NOTE: This wrapper allows one to store persistent data on the GPU memory. */
  class WrapperClassLB2 {
    forwarding var lb2_bound: lb2_bound_data;

    proc init(const jobs: int(32), const machines: int(32))
    {
      this.lb2_bound = new lb2_bound_data(jobs, machines);
    }
  }

  type WrapperLB2 = owned WrapperClassLB2?;

  proc fill_machine_pairs(ref lb2_data: lb2_bound_data/*, enum lb2_variant lb2_type*/): void
  {
    var lb2_type = lb2_variant.LB2_LEARN;

    select (lb2_type) {
      when lb2_variant.LB2_FULL
      {}
      when lb2_variant.LB2_LEARN
      {
        var c: int(32) = 0;
        for i in 0..(lb2_data.nb_machines-2) {
          for j in (i+1)..(lb2_data.nb_machines-1) {
            lb2_data.machine_pairs[0][c] = i;
            lb2_data.machine_pairs[1][c] = j;
            lb2_data.machine_pair_order[c] = c;
            c+=1;
          }
        }
      }
      when lb2_variant.LB2_NABESHIMA
      {
        for i in 0..(lb2_data.nb_machines-2) {
          lb2_data.machine_pairs[0][i] = i;
          lb2_data.machine_pairs[1][i] = i+1;
          lb2_data.machine_pair_order[i] = i;
        }
      }
      when lb2_variant.LB2_LAGEWEG
      {
        for i in 0..(lb2_data.nb_machines-2) {
          lb2_data.machine_pairs[0][i] = i;
          lb2_data.machine_pairs[1][i] = lb2_data.nb_machines-1;
          lb2_data.machine_pair_order[i] = i;
        }
      }
    }
  }

  // term q_iuv in [Lageweg'78]
  proc fill_lags(const lb1_p_times: [] int(32), ref lb2_data: lb2_bound_data): void
  {
    const N = lb2_data.nb_jobs;

    for i in 0..#lb2_data.nb_machine_pairs {
      const m1 = lb2_data.machine_pairs[0][i];
      const m2 = lb2_data.machine_pairs[1][i];

      for j in 0..#N {
        /* lb2.lags[i * N + j] = 0; */
        for k in (m1+1)..(m2-1) {
          lb2_data.lags[i * N + j] += lb1_p_times[k * N + j];
        }
      }
    }
  }

  record johnson_job
  {
    var job: int(32); //job-id
    var partition: int(32); //in partition 0 or 1
    var ptm1: int(32); //processing time on m1
    var ptm2: int(32); //processing time on m2
  }

  // Empty record serves as comparator
  record johnson_comp { }

  //(after partitioning) sorting jobs in ascending order with this comparator yield an optimal schedule for the associated 2-machine FSP [Johnson, S. M. (1954). Optimal two-and three-stage production schedules with setup times included.closed access Naval research logistics quarterly, 1(1), 61–68.]
  proc johnson_comp.compare(j1, j2): int(32)
  {
    /* johnson_job j1 = *((johnson_job*)elem1);
    johnson_job j2 = *((johnson_job*)elem2); */

    //partition 0 before 1
    if (j1.partition == 0 && j2.partition == 1) then return -1;
    if (j1.partition == 1 && j2.partition == 0) then return 1;

    //in partition 0 increasing value of ptm1
    if (j1.partition == 0) {
      if (j2.partition == 1) then return -1;
      return j1.ptm1 - j2.ptm1;
    }
    //in partition 1 decreasing value of ptm1
    if (j1.partition == 1) {
      if (j2.partition == 0) then return 1;
      return j2.ptm2 - j1.ptm2;
    }

    return 0;
  }

  //for each machine-pair (m1,m2), solve 2-machine FSP with processing times
  //  p_1i = PTM[m1][i] + lags[s][i]
  //  p_2i = PTM[m2][i] + lags[s][i]
  //using Johnson's algorithm [Johnson, S. M. (1954). Optimal two-and three-stage production schedules with setup times included.closed access Naval research logistics quarterly, 1(1), 61–68.]
  proc fill_johnson_schedules(const lb1_p_times: [] int(32), ref lb2_data: lb2_bound_data): void
  {
    const N = lb2_data.nb_jobs;
    const ref lags = lb2_data.lags;

    var tmp: [0..#N] johnson_job;

    //for all machine-pairs
    for k in 0..#lb2_data.nb_machine_pairs {
      var m1 = lb2_data.machine_pairs[0][k];
      var m2 = lb2_data.machine_pairs[1][k];

      //partition N jobs into 2 sets {j|p_1j < p_2j} and {j|p_1j >= p_2j}
      for i in 0..#N {
        tmp[i].job = i:int(32);
        tmp[i].ptm1 = lb1_p_times[m1*N + i] + lags[k*N + i];
        tmp[i].ptm2 = lb1_p_times[m2*N + i] + lags[k*N + i];

        if (tmp[i].ptm1 < tmp[i].ptm2) {
          tmp[i].partition = 0;
        } else {
          tmp[i].partition = 1;
        }
      }
      var comp: johnson_comp;
      //sort according to johnson's criterion
      sort(tmp, comp);
      //save optimal schedule for 2-machine problem
      for i in 0..#N {
        lb2_data.johnson_schedules[k*N + i] = tmp[i].job;
      }
    }
  }

  proc set_flags(const permutation, const limit1: int(32), const limit2: int(32), const N: int(32), ref flags): void
  {
    for j in 0..limit1 do
      flags[permutation[j]] = 1;

    for j in limit2..(N-1) do
      flags[permutation[j]] = 1;
  }

  inline proc compute_cmax_johnson(const lb1_p_times: [] int(32), const lb2_data: lb2_bound_data, const flag, ref tmp0: int(32), ref tmp1: int(32), ma0: int(32), ma1: int(32), ind: int(32)): int(32)
  {
    const nb_jobs = lb2_data.nb_jobs;

    use CTypes only c_ptrToConst;
    const lb2_js = c_ptrToConst(lb2_data.johnson_schedules[0]);
    const lb1_pt = c_ptrToConst(lb1_p_times[0]);
    const lb2_lags = c_ptrToConst(lb2_data.lags[0]);

    for j in 0..#nb_jobs {
      var job = lb2_js[ind*nb_jobs + j];
      // j-loop is on unscheduled jobs... (==0 if jobCour is unscheduled)
      if (flag[job] == 0) {
        var ptm0 = lb1_pt[ma0*nb_jobs + job];
        var ptm1 = lb1_pt[ma1*nb_jobs + job];
        var lag = lb2_lags[ind*nb_jobs + job];
        // add job on ma0 and ma1
        tmp0 += ptm0;
        tmp1 = max(tmp1,tmp0 + lag);
        tmp1 += ptm1;
      }
    }

    return tmp1;
  }

  proc lb_makespan(const lb1_p_times: [] int(32), const lb2_data: lb2_bound_data, const flag, const front, const back, const minCmax: int): int(32)
  {
    var lb: int(32) = 0;

    // for all machine-pairs : O(m^2) m*(m-1)/2
    for l in 0..#lb2_data.nb_machine_pairs {
      var i = lb2_data.machine_pair_order[l];

      var ma0 = lb2_data.machine_pairs[0][i];
      var ma1 = lb2_data.machine_pairs[1][i];

      var tmp0 = front[ma0];
      var tmp1 = front[ma1];

      compute_cmax_johnson(lb1_p_times, lb2_data, flag, tmp0, tmp1, ma0, ma1, i);

      tmp1 = max(tmp1 + back[ma1], tmp0 + back[ma0]);

      lb = max(lb,tmp1);

      if (lb > minCmax) {
        break;
      }
    }

    return lb;
  }

  //allows variable nb of machine pairs and get machine pair the realized best lb
  proc lb_makespan_learn(const lb1_p_times: [] int(32), const lb2_data: lb2_bound_data, const flag: [] int(32), const front: [] int(32), const back: [] int(32), const minCmax: int, const nb_pairs: int(32), ref best_index: int(32)): int(32)
  {
    var lb: int(32) = 0;

    for l in 0..#nb_pairs {
      var i = lb2_data.machine_pair_order[l];

      var ma0 = lb2_data.machine_pairs[0][i];
      var ma1 = lb2_data.machine_pairs[1][i];

      var tmp0 = front[ma0];
      var tmp1 = front[ma1];

      compute_cmax_johnson(lb1_p_times, lb2_data, flag, tmp0, tmp1, ma0, ma1, i);

      tmp1 = max(tmp1 + back[ma1], tmp0 + back[ma0]);

      if (tmp1 > lb) {
        best_index = i;
        lb = tmp1;
      }
      // lb=MAX(lb,tmp1);

      if (lb > minCmax) {
        break;
      }
    }

    return lb;
  }

  proc lb2_bound(const lb1_data: lb1_bound_data, const lb2_data: lb2_bound_data, const permutation, const limit1: int(32), const limit2: int(32), const best_cmax: int): int(32)
  {
    /* const N = lb2_data.nb_jobs;
    const M = lb2_data.nb_machines; */

    var front: NUM_MACHINES*int(32);
    var back: NUM_MACHINES* int(32);

    schedule_front(lb1_data, permutation, limit1, front);
    schedule_back(lb1_data, permutation, limit2, back);

    var flags: NUM_JOBS*int(32);
    set_flags(permutation, limit1, limit2, NUM_JOBS, flags);

    return lb_makespan(lb1_data.p_times, lb2_data, flags, front, back, best_cmax);
  }

  proc lb2_children_bounds(const lb1_data: lb1_bound_data, const lb2_data: lb2_bound_data, const permutation, const limit1: int(32), const limit2: int(32), ref lb_begin: [] int(32), ref lb_end: [] int(32), const best_cmax: int, const direction: int): void
  {
    const N = lb1_data.nb_jobs;

    var tmp_perm: [0..#N] int(32);
    for i in 0..#N do tmp_perm[i] = permutation[i];

    select (direction) {
      when -1
       {
        for i in (limit1+1)..(limit2-1) {
          var job = tmp_perm[i];

          tmp_perm[i] <=> tmp_perm[limit1 + 1];
          lb_begin[job] = lb2_bound(lb1_data, lb2_data, tmp_perm, limit1+1, limit2, best_cmax);
          tmp_perm[i] <=> tmp_perm[limit1 + 1];
        }
      }
      when 0
      {
        for i in (limit1+1)..(limit2-1) {
          var job = tmp_perm[i];

          tmp_perm[i] <=> tmp_perm[limit1 + 1];
          lb_begin[job] = lb2_bound(lb1_data, lb2_data, tmp_perm, limit1+1, limit2, best_cmax);
          tmp_perm[i] <=> tmp_perm[limit1 + 1];

          tmp_perm[i] <=> tmp_perm[limit2 - 1];
          lb_end[job] = lb2_bound(lb1_data, lb2_data, tmp_perm, limit1, limit2-1, best_cmax);
          tmp_perm[i] <=> tmp_perm[limit2 - 1];
        }
      }
      when 1
      {
        for i in (limit1+1)..(limit2-1) {
          var job = tmp_perm[i];

          tmp_perm[i] <=> tmp_perm[limit2 - 1];
          lb_end[job] = lb2_bound(lb1_data, lb2_data, tmp_perm, limit1, limit2-1, best_cmax);
          tmp_perm[i] <=> tmp_perm[limit2 - 1];
        }
      }
    }
  }
}

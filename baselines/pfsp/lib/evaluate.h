#ifndef EVALUATE_H
#define EVALUATE_H

#include "PFSP_node.h" // For Nodes definition
#include "c_bound_simple.h" // For structs definitions
#include "c_bound_johnson.h" // For structs definitions

void evaluate_gpu(const int jobs, const int lb, const int size, const int nbBlocks, const int nbBlocks_lb1_d, int* best, const lb1_bound_data lbound1, int* nb_jobs1_d, int* nb_machines1_d, const lb2_bound_data lbound2, Node* parents, int* bounds);

void print_info(const lb1_bound_data* lbound1_d);

void print_info_lb2(const lb2_bound_data* lbound2);

void print_info_lb2_d(const lb2_bound_data lbound2_d);

#endif // EVALUATE_H

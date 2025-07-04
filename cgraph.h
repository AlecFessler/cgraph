#ifndef CGRAPH_H
#define CGRAPH_H

#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CPU_ALIGNMENT 64
#define NO_EDGE 0

#ifdef __cplusplus
extern "C" {
#endif

enum CGRAPH_ERR {
  EOOM,
  EDEVOOM,
  ENOVERT,
  ENOEDGE,
  ENOSLOT,
  NUM_ERRS
};

const char* cgraph_err(const int error);

extern const char* CGRAPH_ERR_STRS[];

struct cgraph {
  int capacity;
  int row_len;
  int *vertices;
  int *matrix;
};

struct cgraph_matrix {
  int row_len;
  int *matrix;
};

int cgraph_matrix_init(struct cgraph_matrix *matrix, const int row_len);
void cgraph_matrix_cleanup(struct cgraph_matrix *matrix);
int cgraph_matrix_at(const struct cgraph_matrix *matrix, const int vertex1, const int vertex2);
void cgraph_print_matrix(const struct cgraph_matrix *matrix);

int round_up_nearest_multiple(const int operand, const int multiple);
int cgraph_init(struct cgraph *graph, const int initial_capacity);
void cgraph_cleanup(struct cgraph *graph);
void cgraph_rand_edges(struct cgraph *graph, const int max_weight, const float edge_prob);
int first_empty_slot(const struct cgraph *graph);
int cgraph_add_vertex(struct cgraph *graph);
int cgraph_remove_vertex(struct cgraph *graph, const int vertex);
int cgraph_add_edge(struct cgraph *graph, const int vertex1, const int vertex2);
int cgraph_add_wedge(struct cgraph *graph, const int vertex1, const int vertex2, const int weight);
int cgraph_remove_edge(struct cgraph *graph, const int vertex1, const int vertex2);
int cgraph_set_weight(struct cgraph *graph, const int vertex1, const int vertex2, const int weight);
int cgraph_has_edge(const struct cgraph *graph, const int vertex1, const int vertex2);
int cgraph_get_weight(const struct cgraph *graph, const int vertex1, const int vertex2);
void cgraph_print(const struct cgraph *graph);

#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>

int cgraph_floyd_warshall(const struct cgraph *graph, struct cgraph_matrix *dist_matrix, struct cgraph_matrix *path_matrix);
__global__ void floyd_warshall_kernel(
  const int *dist_in, const int pitch_dist_in,
  int *dist_out, const int pitch_dist_out,
  const int *path_in, const int pitch_path_in,
  int *path_out, const int pitch_path_out,
  const int k, const int row_len
);

#else

int cgraph_floyd_warshall(const struct cgraph *graph, struct cgraph_matrix *dist_matrix, struct cgraph_matrix *path_matrix);

#endif // CUDA_AVAILABLE

#ifdef __cplusplus
}
#endif

#endif // CGRAPH_H

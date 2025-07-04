#include "cgraph.h"

int main() {
  struct cgraph graph;
  const int capacity = 10;
  cgraph_init(&graph, capacity);

  for (int i = 0; i < capacity; i++) {
    cgraph_add_vertex(&graph);
  }
  cgraph_print(&graph);

  cgraph_rand_edges(&graph, 100, 1.0f);
  cgraph_print(&graph);

  struct cgraph_matrix dist_matrix;
  struct cgraph_matrix path_matrix;
  cgraph_floyd_warshall(&graph, &dist_matrix, &path_matrix);

  cgraph_print_matrix(&dist_matrix);
  cgraph_print_matrix(&path_matrix);

  cgraph_cleanup(&graph);
  cgraph_matrix_cleanup(&dist_matrix);
  cgraph_matrix_cleanup(&path_matrix);
  return 0;
}

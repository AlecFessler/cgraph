#include "cgraph.h"

const char* CGRAPH_ERR_STRS[] = {
  "Out of memory",
  "Gpu out of memory",
  "Vertex doesn't exist",
  "Edge doesn't exist",
  "All vertex slots taken"
};

const char* cgraph_err(const int error) {
  return CGRAPH_ERR_STRS[-error];
}

int cgraph_matrix_init(struct cgraph_matrix *matrix, const int row_len) {
  const int size = row_len * row_len * sizeof(int);
  matrix->matrix = malloc(size);
  if (matrix->matrix == NULL) {
    return -EOOM;
  }
  memset(matrix->matrix, NO_EDGE, size);

  matrix->row_len = row_len;
  return 0;
}

void cgraph_matrix_cleanup(struct cgraph_matrix *matrix) {
  free(matrix->matrix);
}

int cgraph_matrix_at(const struct cgraph_matrix *matrix, const int vertex1, const int vertex2) {
  if (vertex1 >= matrix->row_len || vertex2 >= matrix->row_len) {
    return -ENOVERT;
  }
  const int idx = vertex1 * matrix->row_len + vertex2;
  return matrix->matrix[idx];
}

void cgraph_print_matrix(const struct cgraph_matrix *matrix) {
  for (int i = 0; i < matrix->row_len; i++) {
    if (i > 0)
      printf("\n");
    for (int j = 0; j < matrix->row_len; j++) {
      const int ij = cgraph_matrix_at(matrix, i, j);
      printf("%i ", ij);
    }
  }
  printf("\n\n");
}

int round_up_nearest_multiple(const int operand, const int multiple) {
  return operand + multiple - operand % multiple;
}

int cgraph_init(struct cgraph *graph, const int initial_capacity) {
  const int vert_per_cache = CPU_ALIGNMENT / sizeof(int);
  const int capacity = round_up_nearest_multiple(initial_capacity, vert_per_cache);
  graph->vertices = aligned_alloc(CPU_ALIGNMENT, capacity * sizeof(int));
  if (graph->vertices == NULL) {
    return -EOOM;
  }
  memset(graph->vertices, 0, capacity * sizeof(int));

  const int size = capacity * capacity * sizeof(int);
  graph->matrix = aligned_alloc(CPU_ALIGNMENT, size);
  if (graph->matrix == NULL) {
    return -EOOM;
  }
  memset(graph->matrix, NO_EDGE, size);

  graph->capacity = capacity;
  graph->row_len = 0;

  return 0;
}

void cgraph_cleanup(struct cgraph *graph) {
  free(graph->vertices);
  free(graph->matrix);
}

void cgraph_rand_edges(struct cgraph *graph, const int max_weight, const float edge_prob) {
  for (int i = 0; i < graph->row_len; i++) {
    for (int j = 0; j < graph->row_len; j++) {
      if (i == j) {
        cgraph_add_wedge(graph, i, j, NO_EDGE);
      } else if ((rand() / (float)RAND_MAX) < edge_prob) {
        int weight = rand() % max_weight;
        cgraph_add_wedge(graph, i, j, weight);
      }
    }
  }
}

int first_empty_slot(const struct cgraph *graph) {
  for (int i = 0; i < graph->capacity; i++) {
    if (graph->vertices[i] == 0) {
      return i;
    }
  }
  return -ENOSLOT;
}

int cgraph_add_vertex(struct cgraph *graph) {
  const int first_available = first_empty_slot(graph);

  if (first_available == -ENOSLOT) {
    const int new_capacity = graph->capacity * 2;
    const int vert_per_cache = CPU_ALIGNMENT / sizeof(int);
    assert(new_capacity % vert_per_cache == 0);

    int *new_vertices = aligned_alloc(CPU_ALIGNMENT, new_capacity * sizeof(int));
    if (new_vertices == NULL) {
      return -EOOM;
    }

    for (int i = 0; i < graph->row_len; i++) {
      new_vertices[i] = graph->vertices[i];
    }

    const int size = new_capacity * new_capacity * sizeof(int);
    int *new_matrix = aligned_alloc(CPU_ALIGNMENT, size);
    if (new_matrix == NULL) {
      free(new_vertices);
      return -EOOM;
    }

    for (int i = 0; i < graph->row_len; i++) {
      for (int j = 0; j < graph->row_len; j++) {
        const int old_idx = i * graph->capacity + j;
        const int new_idx = i * new_capacity + j;
        new_matrix[new_idx] = graph->matrix[old_idx];
      }
    }

    free(graph->vertices);
    free(graph->matrix);
    graph->vertices = new_vertices;
    graph->matrix = new_matrix;
    graph->capacity = new_capacity;
  }

  if (first_available == graph->row_len) {
    graph->row_len += 1;
  }

  graph->vertices[first_available] = 1;
  return first_available;
}

int cgraph_remove_vertex(struct cgraph *graph, const int vertex) {
  if (vertex >= graph->row_len) {
    return -ENOVERT;
  }
  if (graph->vertices[vertex] == 0) {
    return -ENOVERT;
  }

  for (int i = 0; i < graph->capacity; i++) {
    const int col_idx = i * graph->capacity + vertex;
    const int row_idx = graph->capacity * vertex + i;
    graph->matrix[col_idx] = 0;
    graph->matrix[row_idx] = 0;
  }

  graph->vertices[vertex] = 0;

  return 0;
}

int cgraph_add_edge(struct cgraph *graph, const int vertex1, const int vertex2) {
  if (vertex1 >= graph->row_len || vertex2 >= graph->row_len) {
    return -ENOVERT;
  }
  if (graph->vertices[vertex1] == 0 || graph->vertices[vertex2] == 0) {
    return -ENOVERT;
  }

  const int idx = vertex1 * graph->capacity + vertex2;
  graph->matrix[idx] = 1;

  return 0;
}

int cgraph_add_wedge(struct cgraph *graph, const int vertex1, const int vertex2, const int weight) {
  if (vertex1 >= graph->row_len || vertex2 >= graph->row_len) {
    return -ENOVERT;
  }
  if (graph->vertices[vertex1] == 0 || graph->vertices[vertex2] == 0) {
    return -ENOVERT;
  }

  const int idx = vertex1 * graph->capacity + vertex2;
  graph->matrix[idx] = weight;

  return 0;
}

int cgraph_remove_edge(struct cgraph *graph, const int vertex1, const int vertex2) {
  if (vertex1 >= graph->row_len || vertex2 >= graph->row_len) {
    return -ENOVERT;
  }
  if (graph->vertices[vertex1] == 0 || graph->vertices[vertex2] == 0) {
    return -ENOVERT;
  }

  const int idx = vertex1 * graph->capacity + vertex2;

  if (graph->matrix[idx] == NO_EDGE) {
    return -ENOEDGE;
  }

  graph->matrix[idx] = 0;

  return 0;
}

int cgraph_set_weight(struct cgraph *graph, const int vertex1, const int vertex2, const int weight) {
  if (vertex1 >= graph->row_len || vertex2 >= graph->row_len) {
    return -ENOVERT;
  }
  if (graph->vertices[vertex1] == 0 || graph->vertices[vertex2] == 0) {
    return -ENOVERT;
  }

  const int idx = vertex1 * graph->capacity + vertex2;

  if (graph->matrix[idx] == NO_EDGE) {
    return -ENOEDGE;
  }

  graph->matrix[idx] = weight;

  return 0;
}

int cgraph_has_edge(const struct cgraph *graph, const int vertex1, const int vertex2) {
  if (vertex1 >= graph->row_len || vertex2 >= graph->row_len) {
    return -ENOVERT;
  }
  if (graph->vertices[vertex1] == 0 || graph->vertices[vertex2] == 0) {
    return -ENOVERT;
  }

  const int idx = vertex1 * graph->capacity + vertex2;
  return graph->matrix[idx];
}

int cgraph_get_weight(const struct cgraph *graph, const int vertex1, const int vertex2) {
  if (vertex1 >= graph->row_len || vertex2 >= graph->row_len) {
    return -ENOVERT;
  }
  if (graph->vertices[vertex1] == 0 || graph->vertices[vertex2] == 0) {
    return -ENOVERT;
  }

  const int idx = vertex1 * graph->capacity + vertex2;

  if (graph->matrix[idx] == NO_EDGE) {
    return -ENOEDGE;
  }

  return graph->matrix[idx];
}

void cgraph_print(const struct cgraph *graph) {
  for (int i = 0; i < graph->row_len; i++) {
    if (i > 0)
      printf("\n");
    for (int j = 0; j < graph->row_len; j++) {
      const int idx = i * graph->capacity + j;
      const int ij = graph->matrix[idx];
      printf("%i ", ij);
    }
  }
  printf("\n\n");
}

#ifndef CUDA_AVAILABLE

int cgraph_floyd_warshall(const struct cgraph *graph, struct cgraph_matrix *dist_matrix, struct cgraph_matrix *path_matrix) {
  const int row_len = graph->row_len;

  int ret = 0;
  ret = cgraph_matrix_init(dist_matrix, row_len);
  if (ret != 0) {
    return ret;
  }

  ret = cgraph_matrix_init(path_matrix, row_len);
  if (ret != 0) {
    cgraph_matrix_cleanup(dist_matrix);
    return ret;
  }

  dist_matrix->row_len = row_len;
  path_matrix->row_len = row_len;

  for (int i = 0; i < row_len; i++) {
    for (int j = 0; j < row_len; j++) {
      const int dist_idx = i * row_len + j;
      const int graph_idx = i * graph->capacity + j;
      dist_matrix->matrix[dist_idx] = graph->matrix[graph_idx];
    }
  }

  for (int i = 0; i < row_len; i++) {
    for (int j = 0; j < row_len; j++) {
      const int path_idx = i * row_len + j;
      const int graph_idx = i * graph->capacity + j;
      if (i == j) {
        path_matrix->matrix[path_idx] = NO_EDGE;
      } else if (graph->matrix[graph_idx] != NO_EDGE) {
        path_matrix->matrix[path_idx] = j;
      } else {
        path_matrix->matrix[path_idx] = NO_EDGE;
      }
    }
  }

  for (int k = 0; k < row_len; k++) {
    if (!graph->vertices[k])
      continue;

    for (int i = 0; i < row_len; i++) {
      if (!graph->vertices[i])
        continue;

      const int ik_idx = i * row_len + k;
      if (dist_matrix->matrix[ik_idx] == NO_EDGE)
        continue;

      for (int j = 0; j < row_len; j++) {
        if (!graph->vertices[j])
          continue;

        const int kj_idx = k * row_len + j;
        if (dist_matrix->matrix[kj_idx] == NO_EDGE)
          continue;

        const int ij_idx = i * row_len + j;
        const int ik = dist_matrix->matrix[ik_idx];
        const int kj = dist_matrix->matrix[kj_idx];
        const int ij = dist_matrix->matrix[ij_idx];

        if (ij > ik + kj) {
          dist_matrix->matrix[ij_idx] = ik + kj;
          path_matrix->matrix[ij_idx] = path_matrix->matrix[ik_idx];
        }
      }
    }
  }

  return 0;
}

#endif // CUDA_AVAILABLE

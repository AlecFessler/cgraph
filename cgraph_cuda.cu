#include "cgraph.h"

__global__ void floyd_warshall_kernel(
  const int *dist_in, const int pitch_dist_in,
  int *dist_out, const int pitch_dist_out,
  const int *path_in, const int pitch_path_in,
  int *path_out, const int pitch_path_out,
  const int k, const int row_len
) {
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= row_len || j >= row_len)
    return;

  const int ik = dist_in[i * pitch_dist_in + k];
  const int kj = dist_in[k * pitch_dist_in + j];
  const int ij = dist_in[i * pitch_dist_in + j];
  const int ikj = ik + kj;

  const int use_new = (ik != NO_EDGE && kj != NO_EDGE && ikj < ij);
  dist_out[i * pitch_dist_out + j] = use_new * ikj + (1 - use_new) * ij;
  path_out[i * pitch_path_out + j] = use_new * path_in[i * pitch_path_in + k] + (1 - use_new) * path_in[i * pitch_path_in + j];
}

int cgraph_floyd_warshall(const struct cgraph *graph,
                          struct cgraph_matrix *dist_matrix,
                          struct cgraph_matrix *path_matrix) {
  const int row_len = graph->row_len;
  const int size = row_len * sizeof(int);

  int ret = cgraph_matrix_init(dist_matrix, row_len);
  if (ret != 0) return ret;

  ret = cgraph_matrix_init(path_matrix, row_len);
  if (ret != 0) {
    cgraph_matrix_cleanup(dist_matrix);
    return ret;
  }

  dist_matrix->row_len = row_len;
  path_matrix->row_len = row_len;

  int *dev_dist_in, *dev_dist_out;
  int *dev_path_in, *dev_path_out;
  size_t pitch_dist_in_bytes, pitch_dist_out_bytes;
  size_t pitch_path_in_bytes, pitch_path_out_bytes;

  cudaMallocPitch((void**)&dev_dist_in, &pitch_dist_in_bytes, size, row_len);
  cudaMallocPitch((void**)&dev_dist_out, &pitch_dist_out_bytes, size, row_len);
  cudaMallocPitch((void**)&dev_path_in, &pitch_path_in_bytes, size, row_len);
  cudaMallocPitch((void**)&dev_path_out, &pitch_path_out_bytes, size, row_len);

  cudaMemcpy2D(dev_dist_in, pitch_dist_in_bytes,
               graph->matrix, graph->capacity * sizeof(int),
               size, row_len, cudaMemcpyHostToDevice);

  cudaMemcpy2D(dev_dist_out, pitch_dist_out_bytes,
               dev_dist_in, pitch_dist_in_bytes,
               size, row_len, cudaMemcpyDeviceToDevice);

  for (int i = 0; i < row_len; i++) {
    for (int j = 0; j < row_len; j++) {
      int path_idx = i * row_len + j;
      int graph_idx = i * graph->capacity + j;
      if (i == j) {
        path_matrix->matrix[path_idx] = NO_EDGE;
      } else if (graph->matrix[graph_idx] != NO_EDGE) {
        path_matrix->matrix[path_idx] = j;
      } else {
        path_matrix->matrix[path_idx] = NO_EDGE;
      }
    }
  }

  cudaMemcpy2D(dev_path_in, pitch_path_in_bytes,
               path_matrix->matrix, size,
               size, row_len, cudaMemcpyHostToDevice);

  cudaMemcpy2D(dev_path_out, pitch_path_out_bytes,
               dev_path_in, pitch_path_in_bytes,
               size, row_len, cudaMemcpyDeviceToDevice);

  int pitch_dist_in = pitch_dist_in_bytes / sizeof(int);
  int pitch_dist_out = pitch_dist_out_bytes / sizeof(int);
  int pitch_path_in = pitch_path_in_bytes / sizeof(int);
  int pitch_path_out = pitch_path_out_bytes / sizeof(int);

  const int block_size = 16;
  dim3 block_dim(block_size, block_size);
  int grid_size = (row_len + block_size - 1) / block_size;
  dim3 grid_dim(grid_size, grid_size);

  for (int k = 0; k < row_len; k++) {
    floyd_warshall_kernel<<<grid_dim, block_dim>>>(
      dev_dist_in, pitch_dist_in,
      dev_dist_out, pitch_dist_out,
      dev_path_in, pitch_path_in,
      dev_path_out, pitch_path_out,
      k, row_len);

    cudaDeviceSynchronize();

    int *tmp_ptr;
    int tmp_pitch;

    tmp_ptr = dev_dist_in; dev_dist_in = dev_dist_out; dev_dist_out = tmp_ptr;
    tmp_pitch = pitch_dist_in; pitch_dist_in = pitch_dist_out; pitch_dist_out = tmp_pitch;

    tmp_ptr = dev_path_in; dev_path_in = dev_path_out; dev_path_out = tmp_ptr;
    tmp_pitch = pitch_path_in; pitch_path_in = pitch_path_out; pitch_path_out = tmp_pitch;
  }

  cudaMemcpy2D(dist_matrix->matrix, size,
               dev_dist_in, pitch_dist_in_bytes,
               size, row_len, cudaMemcpyDeviceToHost);

  cudaMemcpy2D(path_matrix->matrix, size,
               dev_path_in, pitch_path_in_bytes,
               size, row_len, cudaMemcpyDeviceToHost);

  cudaFree(dev_dist_in);
  cudaFree(dev_dist_out);
  cudaFree(dev_path_in);
  cudaFree(dev_path_out);

  return 0;
}

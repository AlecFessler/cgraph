# CGRAPH: A High-Performance CPU/GPU Graph Library

**CGRAPH** is a graph processing library written in C with optional CUDA acceleration. It supports both CPU and GPU execution of the Floyd–Warshall algorithm, random graph generation, and efficient memory layout using cache-line aligned storage.

## 🚀 Features

- Matrix-based graph representation
- Cache-aligned memory with `aligned_alloc`
- Dynamic vertex addition and automatic resizing
- Configurable random edge generation
- Error code handling with descriptive messages
- Dual-mode Floyd–Warshall algorithm:
  - Pure C implementation (fallback)
  - CUDA-accelerated implementation with `cudaMallocPitch` and `cudaMemcpy2D`
- Branchless CUDA kernel for warp-efficiency and coalesced memory access
- Optional CUDA detection in Makefile

## 📦 Build Instructions

### Requirements

- GCC (or compatible C compiler)
- Optional: CUDA Toolkit (for GPU acceleration)

### Compile

```bash
make
```

The Makefile will automatically detect if `nvcc` is installed and compile the CUDA components accordingly.

### Clean

```bash
make clean
```

## 🧪 Example Usage

```c
#include "cgraph.h"

int main() {
  struct cgraph graph;
  cgraph_init(&graph, 10);

  for (int i = 0; i < 10; i++) {
    cgraph_add_vertex(&graph);
  }

  cgraph_rand_edges(&graph, 100, 1.0f);
  cgraph_print(&graph);

  struct cgraph_matrix dist, path;
  cgraph_floyd_warshall(&graph, &dist, &path);

  cgraph_print_matrix(&dist);
  cgraph_print_matrix(&path);

  cgraph_cleanup(&graph);
  cgraph_matrix_cleanup(&dist);
  cgraph_matrix_cleanup(&path);

  return 0;
}
```

## 🧠 Planned Features

- Path reconstruction
- Graph serialization (save/load)
- Support for undirected graphs
- Float-weighted edges
- CLI tools for graph inspection and benchmarks

## 📄 License

MIT License

---

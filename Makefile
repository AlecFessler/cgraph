TARGET := cgraph
C_SOURCES := main.c cgraph.c
CUDA_SOURCES := cgraph_cuda.cu

CUDA_PATH ?= /opt/cuda
NVCC := $(CUDA_PATH)/bin/nvcc

C_OBJECTS := $(C_SOURCES:.c=.o)
CUDA_OBJECTS := $(CUDA_SOURCES:.cu=.o)
CUDA_DLINK := cuda_dlink.o

HAVE_CUDA := $(shell test -x $(NVCC) && echo 1)

ifeq ($(HAVE_CUDA),1)
    OBJS := $(C_OBJECTS) $(CUDA_OBJECTS) $(CUDA_DLINK)
    LD := $(NVCC)
    CFLAGS := -Wall -Wextra -std=c11 -g -DCUDA_AVAILABLE -I$(CUDA_PATH)/include
    NVCCFLAGS := -std=c++17 -g -dc -I$(CUDA_PATH)/include -Xcompiler "-Wall -Wextra -g" -DCUDA_AVAILABLE
    LDLIBS := -L$(CUDA_PATH)/lib64 -lcudart
else
    OBJS := $(C_OBJECTS)
    LD := gcc
    CFLAGS := -Wall -Wextra -std=c11 -g
    LDLIBS :=
endif

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(LD) -o $@ $^ $(LDLIBS)

%.o: %.c
	gcc -x c $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

cuda_dlink.o: $(CUDA_OBJECTS)
	$(NVCC) -dlink -o $@ $^ -Xcompiler "-Wall -Wextra -g"

clean:
	rm -f $(TARGET) *.o

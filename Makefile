CC ?= cc

BUILD_DIR := build
INCLUDE_DIR := include
SRC_DIR := src
TEST_DIR := tests

BAK_DEST ?= /home/rick/Remote/Terrastation/stuff/ATT-1

CFLAGS := -std=c11 -Wall -Wextra -Wpedantic -Werror -pthread -I$(INCLUDE_DIR)
LDFLAGS :=
LDLIBS := -lm -pthread
CUDA ?= 0
CUDA_HOME ?= /usr/local/cuda

ifeq ($(CUDA),1)
CFLAGS += -DATT1_ENABLE_CUDA -I$(CUDA_HOME)/include
LDFLAGS += -L$(CUDA_HOME)/lib64
LDLIBS += -lcudart -lcublas
endif

SIM_BIN := $(BUILD_DIR)/att1-sim
INSPECT_BIN := $(BUILD_DIR)/att1-inspect
BENCH_BIN := $(BUILD_DIR)/att1-bench
Q8_BENCH_BIN := $(BUILD_DIR)/att1-q8-bench
SIZE_BIN := $(BUILD_DIR)/att1-size
TINY_LLM_BIN := $(BUILD_DIR)/run_tiny_llm
CLUSTER_LLM_BIN := $(BUILD_DIR)/run_cluster_llm
TEST_NAMES := smoke tensor matmul rmsnorm softmax rope silu swiglu \
	kv_cache attention transformer_block kv_mmu fabric runtime model_loader \
	tokenizer sampler infer shard shard_meta shard_meta_fixture shard_meta_report shard_meta_consistency shard_meta_plan cluster_infer trace quant matmul_q8 backend bench_smoke \
	cuda_matmul cuda_norm cuda_ffn cuda_rope cuda_attention cuda_transformer_block cuda_infer cuda_cluster cuda_bench \
	q8_bench q8_cluster cuda_q8_cluster backend_matrix
TEST_BINS := $(addprefix $(BUILD_DIR)/test_,$(TEST_NAMES))

COMMON_SRCS := \
	$(SRC_DIR)/log.c \
	$(SRC_DIR)/tensor.c \
	$(SRC_DIR)/matmul.c \
	$(SRC_DIR)/norm.c \
	$(SRC_DIR)/rope.c \
	$(SRC_DIR)/ffn.c \
	$(SRC_DIR)/tokenizer.c \
	$(SRC_DIR)/sampler.c \
	$(SRC_DIR)/backend_cpu_f32.c \
	$(SRC_DIR)/backend_cpu_q8.c \
	$(SRC_DIR)/backend_cuda.c \
	$(SRC_DIR)/quant.c \
	$(SRC_DIR)/matmul_q8.c \
	$(SRC_DIR)/trace.c \
	$(SRC_DIR)/model_view.c \
	$(SRC_DIR)/infer.c \
	$(SRC_DIR)/shard.c \
	$(SRC_DIR)/cluster_infer.c \
	$(SRC_DIR)/fabric.c \
	simulator/sim_fabric_bus.c \
	$(SRC_DIR)/tile.c \
	$(SRC_DIR)/runtime.c \
	simulator/sim_tile_thread.c \
	$(SRC_DIR)/kv_cache.c \
	$(SRC_DIR)/kv_mmu.c \
	$(SRC_DIR)/model_loader.c \
	$(SRC_DIR)/shard_meta.c \
	$(SRC_DIR)/attention.c \
	$(SRC_DIR)/transformer_block.c

COMMON_OBJS := $(patsubst %.c,$(BUILD_DIR)/%.o,$(COMMON_SRCS))
SIM_OBJS := $(BUILD_DIR)/$(SRC_DIR)/main.o $(COMMON_OBJS)
INSPECT_OBJS := $(BUILD_DIR)/tools/att1-inspect.o $(COMMON_OBJS)
BENCH_OBJS := $(BUILD_DIR)/tools/att1-bench.o $(COMMON_OBJS)
Q8_BENCH_OBJS := $(BUILD_DIR)/tools/att1-q8-bench.o $(COMMON_OBJS)
SIZE_OBJS := $(BUILD_DIR)/tools/att1-size.o $(COMMON_OBJS)
TINY_LLM_OBJS := $(BUILD_DIR)/examples/run_tiny_llm.o $(COMMON_OBJS)
CLUSTER_LLM_OBJS := $(BUILD_DIR)/examples/run_cluster_llm.o $(COMMON_OBJS)

.PHONY: all clean test bak restore
.SECONDARY:

all: $(SIM_BIN) $(INSPECT_BIN) $(BENCH_BIN) $(Q8_BENCH_BIN) $(SIZE_BIN) $(TINY_LLM_BIN) $(CLUSTER_LLM_BIN)

$(SIM_BIN): $(SIM_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(INSPECT_BIN): $(INSPECT_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BENCH_BIN): $(BENCH_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(Q8_BENCH_BIN): $(Q8_BENCH_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(SIZE_BIN): $(SIZE_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(TINY_LLM_BIN): $(TINY_LLM_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(CLUSTER_LLM_BIN): $(CLUSTER_LLM_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD_DIR)/test_%: $(BUILD_DIR)/$(TEST_DIR)/test_%.o $(COMMON_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c -o $@ $<

test: $(INSPECT_BIN) $(BENCH_BIN) $(Q8_BENCH_BIN) $(SIZE_BIN) $(TEST_BINS)
	@for test_bin in $(TEST_BINS); do \
		./$$test_bin || exit $$?; \
	done

bak:
	@echo "Backing up ATT-1 to $(BAK_DEST)"
	@mkdir -p "$(BAK_DEST)"
	rsync -aH --info=progress2 \
		--exclude='$(BUILD_DIR)/' \
		--exclude='cmake-build-*/' \
		--exclude='out/' \
		--exclude='dist/' \
		--exclude='*.o' \
		--exclude='*.a' \
		--exclude='*.so' \
		--exclude='*.dylib' \
		--exclude='*.dll' \
		--exclude='*.exe' \
		--exclude='*.pyc' \
		--exclude='__pycache__/' \
		--exclude='.pytest_cache/' \
		--exclude='.cache/' \
		--exclude='.venv/' \
		--exclude='venv/' \
		--exclude='models/*.bin' \
		--exclude='models/*.gguf' \
		--exclude='models/*.safetensors' \
		./ "$(BAK_DEST)/"


restore:
	@echo "Restoring ATT-1 from $(BAK_DEST) to current working tree"
	@test -d "$(BAK_DEST)" || { echo "Backup source not found: $(BAK_DEST)"; exit 1; }
	rsync -aH --delete --info=progress2 \
		--exclude='$(BUILD_DIR)/' \
		--exclude='cmake-build-*/' \
		--exclude='out/' \
		--exclude='dist/' \
		--exclude='*.o' \
		--exclude='*.a' \
		--exclude='*.so' \
		--exclude='*.dylib' \
		--exclude='*.dll' \
		--exclude='*.exe' \
		--exclude='*.pyc' \
		--exclude='__pycache__/' \
		--exclude='.pytest_cache/' \
		--exclude='.cache/' \
		--exclude='.venv/' \
		--exclude='venv/' \
		--exclude='models/*.bin' \
		--exclude='models/*.gguf' \
		--exclude='models/*.safetensors' \
		"$(BAK_DEST)/" ./

clean:
	rm -rf $(BUILD_DIR)

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
AIMU_REPLAY_BIN := $(BUILD_DIR)/att1-aimu-replay
AIMU_MMIO_REPLAY_BIN := $(BUILD_DIR)/att1-aimu-mmio-replay
TINY_LLM_BIN := $(BUILD_DIR)/run_tiny_llm
CLUSTER_LLM_BIN := $(BUILD_DIR)/run_cluster_llm
TEST_NAMES := smoke tensor matmul rmsnorm softmax rope silu swiglu \
	kv_cache attention transformer_block kv_mmu fabric runtime model_loader \
	tokenizer sampler infer shard shard_meta shard_meta_fixture shard_meta_report shard_meta_consistency shard_meta_plan shard_meta_exec cluster_infer trace quant matmul_q8 backend bench_smoke tok_meta tok_meta_select tok_ext \
	cuda_matmul cuda_norm cuda_ffn cuda_rope cuda_attention cuda_transformer_block cuda_infer cuda_cluster cuda_bench \
     q8_bench q8_cluster cuda_q8_cluster backend_matrix converter_validation quant_q4 quant_q4_pack matmul_q4 quant_q4_fixture infer_q4 cluster_infer_q4 q4_bench cuda_matmul_q4 cuda_infer_q4 cuda_cluster_infer_q4 \
     aimu_cmdq aimu_device aimu_dma aimu_trace aimu_mmio aimu_host aimu_userspace aimu_mmio_replay aimu_mem \
     aimu_exec aimu_conformance \
     aimu_mmio_regression aimu_endpoint backend_pcie
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
	$(SRC_DIR)/backend_cpu_q4.c \
	$(SRC_DIR)/backend_cuda.c \
	$(SRC_DIR)/backend_pcie.c \
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
	$(SRC_DIR)/tok_meta.c \
	$(SRC_DIR)/tok_ext.c \
	$(SRC_DIR)/attention.c \
	$(SRC_DIR)/transformer_block.c \
	$(SRC_DIR)/aimu_cmdq.c \
	$(SRC_DIR)/aimu_device.c \
	$(SRC_DIR)/aimu_dma.c \
	$(SRC_DIR)/aimu_trace.c \
$(SRC_DIR)/aimu_mmio.c \
     $(SRC_DIR)/aimu_host.c \
	$(SRC_DIR)/aimu_userspace.c \
	$(SRC_DIR)/aimu_mem.c \
	$(SRC_DIR)/aimu_exec.c \
	$(SRC_DIR)/aimu_conformance.c \
	$(SRC_DIR)/aimu_endpoint_protocol.c \
	$(SRC_DIR)/aimu_endpoint_client.c

COMMON_OBJS := $(patsubst %.c,$(BUILD_DIR)/%.o,$(COMMON_SRCS))
SIM_OBJS := $(BUILD_DIR)/$(SRC_DIR)/main.o $(COMMON_OBJS)
INSPECT_OBJS := $(BUILD_DIR)/tools/att1-inspect.o $(COMMON_OBJS)
BENCH_OBJS := $(BUILD_DIR)/tools/att1-bench.o $(COMMON_OBJS)
Q8_BENCH_OBJS := $(BUILD_DIR)/tools/att1-q8-bench.o $(COMMON_OBJS)
SIZE_OBJS := $(BUILD_DIR)/tools/att1-size.o $(COMMON_OBJS)
AIMU_REPLAY_OBJS := $(BUILD_DIR)/tools/att1-aimu-replay.o $(COMMON_OBJS)
AIMU_EMULATOR_BIN := $(BUILD_DIR)/att1-aimu-mmio-emulator
AIMU_EMULATOR_OBJS := $(BUILD_DIR)/tools/att1-aimu-mmio-emulator.o $(COMMON_OBJS)
AIMU_MMIO_REPLAY_OBJS := $(BUILD_DIR)/tools/att1-aimu-mmio-replay.o $(COMMON_OBJS)
AIMU_ENDPOINT_BIN := $(BUILD_DIR)/att1-aimu-endpoint
AIMU_ENDPOINT_OBJS := $(BUILD_DIR)/tools/att1-aimu-endpoint.o $(COMMON_OBJS)
TINY_LLM_OBJS := $(BUILD_DIR)/examples/run_tiny_llm.o $(COMMON_OBJS)
CLUSTER_LLM_OBJS := $(BUILD_DIR)/examples/run_cluster_llm.o $(COMMON_OBJS)

.PHONY: all clean test test-verbose regression bak restore asan ubsan sanitizer test-asan test-ubsan clean-asan clean-ubsan clean-sanitizer fuzz-loader fuzz-json fuzz-coverage fuzz-smoke fuzz-libfuzzer fuzz-afl clean-fuzz docs-check
.SECONDARY:

all: $(SIM_BIN) $(INSPECT_BIN) $(BENCH_BIN) $(Q8_BENCH_BIN) $(SIZE_BIN) $(AIMU_REPLAY_BIN) $(AIMU_EMULATOR_BIN) $(AIMU_MMIO_REPLAY_BIN) $(AIMU_ENDPOINT_BIN) $(TINY_LLM_BIN) $(CLUSTER_LLM_BIN)

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

$(AIMU_REPLAY_BIN): $(AIMU_REPLAY_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(AIMU_EMULATOR_BIN): $(AIMU_EMULATOR_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(AIMU_MMIO_REPLAY_BIN): $(AIMU_MMIO_REPLAY_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(AIMU_ENDPOINT_BIN): $(AIMU_ENDPOINT_OBJS)
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

test: $(INSPECT_BIN) $(BENCH_BIN) $(Q8_BENCH_BIN) $(SIZE_BIN) $(AIMU_REPLAY_BIN) $(AIMU_EMULATOR_BIN) $(AIMU_MMIO_REPLAY_BIN) $(AIMU_ENDPOINT_BIN) $(TEST_BINS)
	@for test_bin in $(TEST_BINS); do \
		./$$test_bin || exit $$?; \
	done

test-verbose: $(INSPECT_BIN) $(BENCH_BIN) $(Q8_BENCH_BIN) $(SIZE_BIN) $(AIMU_REPLAY_BIN) $(AIMU_EMULATOR_BIN) $(AIMU_MMIO_REPLAY_BIN) $(AIMU_ENDPOINT_BIN) $(TEST_BINS)
	@for test_bin in $(TEST_BINS); do \
		printf '\n=== %s ===\n' "$$test_bin"; \
		./$$test_bin || exit $$?; \
	done

# M136: deterministic local regression runner (CPU-only by default)
# Runs: make clean, make, make test, then all Python validation layers.
# Pass --cuda for CUDA signoff (requires RTX 3090-class host).
# Pass --report-json FILE to write a machine-readable summary.
regression:
	python3 compiler/run_full_regression.py

bak:
	@echo "Backing up ATT-1 to $(BAK_DEST)"
	@mkdir -p "$(BAK_DEST)"
	rsync -aHL --info=progress2 \
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
	rsync -aHL --delete --info=progress2 \
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

# M142: ASAN / UBSAN sanitizer build targets (CPU-only, separate build dirs)
# These targets use dedicated build directories so sanitizer artifacts never
# contaminate the normal build/.
# Usage:
#   make clean && make test-asan    # AddressSanitizer
#   make clean && make test-ubsan   # UndefinedBehaviorSanitizer
#   make sanitizer                  # build both (does not run tests)
#   make clean-sanitizer            # remove build-asan/ and build-ubsan/
BUILD_ASAN  := build-asan
BUILD_UBSAN := build-ubsan
ASAN_FLAGS  := -fsanitize=address -fno-omit-frame-pointer -g
UBSAN_FLAGS := -fsanitize=undefined -fno-omit-frame-pointer -g

asan:
	$(MAKE) CUDA=0 BUILD_DIR=$(BUILD_ASAN) CFLAGS="$(CFLAGS) $(ASAN_FLAGS)" all

ubsan:
	$(MAKE) CUDA=0 BUILD_DIR=$(BUILD_UBSAN) CFLAGS="$(CFLAGS) $(UBSAN_FLAGS)" all

sanitizer: asan ubsan

test-asan:
	$(MAKE) CUDA=0 all
	$(MAKE) CUDA=0 BUILD_DIR=$(BUILD_ASAN) CFLAGS="$(CFLAGS) $(ASAN_FLAGS)" test

test-ubsan:
	$(MAKE) CUDA=0 all
	$(MAKE) CUDA=0 BUILD_DIR=$(BUILD_UBSAN) CFLAGS="$(CFLAGS) $(UBSAN_FLAGS)" test

clean-asan:
	rm -rf $(BUILD_ASAN)

clean-ubsan:
	rm -rf $(BUILD_UBSAN)

clean-sanitizer: clean-asan clean-ubsan

# M143/M152: Fuzz/smoke harness targets (CPU-only, local hardening only).
# These are deterministic smoke tests that verify the binary model loader and
# JSON schema validators correctly reject malformed/hostile input.
# Usage:
#   make fuzz-loader            # build and run C binary loader harness
#   make fuzz-json              # run Python JSON schema mutation harness
#   make fuzz-coverage          # report deterministic fuzz corpus coverage
#   make fuzz-smoke             # run all deterministic fuzz checks
#   make fuzz-libfuzzer         # optional guided fuzzer binary (requires clang)
#   make fuzz-afl               # optional AFL-compatible binary (requires AFL_CC)
FUZZ_LOADER_BIN  := $(BUILD_DIR)/fuzz_model_loader
FUZZ_LOADER_OBJS := $(BUILD_DIR)/$(TEST_DIR)/fuzz_model_loader.o $(COMMON_OBJS)
FUZZ_CC ?= clang
AFL_CC ?= afl-clang-fast
FUZZ_BUILD_DIR := build-fuzz
FUZZ_AFL_BUILD_DIR := build-afl
FUZZ_LIBFUZZER_BIN := $(FUZZ_BUILD_DIR)/fuzz_model_loader_libfuzzer
FUZZ_AFL_BIN := $(FUZZ_AFL_BUILD_DIR)/fuzz_model_loader_afl
FUZZ_GUIDED_SRC := $(TEST_DIR)/fuzz_model_loader_guided.c
FUZZ_LIBFUZZER_COMMON_OBJS := $(patsubst %.c,$(FUZZ_BUILD_DIR)/%.o,$(COMMON_SRCS))
FUZZ_LIBFUZZER_OBJS := $(FUZZ_BUILD_DIR)/$(FUZZ_GUIDED_SRC:.c=.o) $(FUZZ_LIBFUZZER_COMMON_OBJS)
FUZZ_AFL_COMMON_OBJS := $(patsubst %.c,$(FUZZ_AFL_BUILD_DIR)/%.o,$(COMMON_SRCS))
FUZZ_AFL_OBJS := $(FUZZ_AFL_BUILD_DIR)/$(FUZZ_GUIDED_SRC:.c=.o) $(FUZZ_AFL_COMMON_OBJS)
FUZZ_LIBFUZZER_CFLAGS := $(CFLAGS) -DATT1_LIBFUZZER -fsanitize=fuzzer-no-link,address,undefined
FUZZ_LIBFUZZER_LDFLAGS := $(LDFLAGS) -fsanitize=fuzzer,address,undefined
FUZZ_AFL_CFLAGS := $(CFLAGS)

$(FUZZ_LOADER_BIN): $(FUZZ_LOADER_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(FUZZ_BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(FUZZ_CC) $(FUZZ_LIBFUZZER_CFLAGS) -c -o $@ $<

$(FUZZ_LIBFUZZER_BIN): $(FUZZ_LIBFUZZER_OBJS)
	$(FUZZ_CC) $(FUZZ_LIBFUZZER_LDFLAGS) -o $@ $^ $(LDLIBS)

$(FUZZ_AFL_BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(AFL_CC) $(FUZZ_AFL_CFLAGS) -c -o $@ $<

$(FUZZ_AFL_BIN): $(FUZZ_AFL_OBJS)
	$(AFL_CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

fuzz-loader: $(FUZZ_LOADER_BIN)
	$(FUZZ_LOADER_BIN)

fuzz-json:
	python3 compiler/fuzz_json_schemas.py

fuzz-coverage:
	python3 compiler/report_fuzz_coverage.py

fuzz-smoke: fuzz-loader fuzz-json fuzz-coverage

fuzz-libfuzzer:
	@if ! command -v $(FUZZ_CC) >/dev/null 2>&1; then \
		echo "SKIP: fuzz-libfuzzer requires $(FUZZ_CC)"; \
	else \
		$(MAKE) $(FUZZ_LIBFUZZER_BIN); \
		echo "Built $(FUZZ_LIBFUZZER_BIN)"; \
		echo "Example: $(FUZZ_LIBFUZZER_BIN) -runs=1000 compiler/fixtures/fuzz-seeds"; \
	fi

fuzz-afl:
	@if ! command -v $(AFL_CC) >/dev/null 2>&1; then \
		echo "SKIP: fuzz-afl requires $(AFL_CC)"; \
	else \
		$(MAKE) $(FUZZ_AFL_BIN); \
		echo "Built $(FUZZ_AFL_BIN)"; \
		echo "Example: afl-fuzz -i compiler/fixtures/fuzz-seeds -o /tmp/att1-afl -- $(FUZZ_AFL_BIN) @@"; \
	fi

clean-fuzz:
	rm -rf $(FUZZ_BUILD_DIR) $(FUZZ_AFL_BUILD_DIR)

# M149: Documentation lint and link checker.
# Validates internal Markdown links, required docs, forbidden patterns,
# and milestone/status consistency. CPU-only, no network, no CUDA.
# Usage:
#   make docs-check
#   make docs-check DOCS_CHECK_ARGS="--warn-anchors"
#   make docs-check DOCS_CHECK_ARGS="--report-json /tmp/docs_report.json"
DOCS_CHECK_ARGS ?=
docs-check:
	python3 compiler/check_docs.py $(DOCS_CHECK_ARGS)

clean:
	rm -rf $(BUILD_DIR)

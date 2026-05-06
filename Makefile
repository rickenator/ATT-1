CC ?= cc

BUILD_DIR := build
INCLUDE_DIR := include
SRC_DIR := src
TEST_DIR := tests

CFLAGS := -std=c11 -Wall -Wextra -Wpedantic -Werror -I$(INCLUDE_DIR)
LDFLAGS :=
LDLIBS := -lm

SIM_BIN := $(BUILD_DIR)/att1-sim
TEST_NAMES := smoke tensor matmul rmsnorm softmax rope silu swiglu \
	kv_cache attention transformer_block
TEST_BINS := $(addprefix $(BUILD_DIR)/test_,$(TEST_NAMES))

COMMON_SRCS := \
	$(SRC_DIR)/log.c \
	$(SRC_DIR)/tensor.c \
	$(SRC_DIR)/matmul.c \
	$(SRC_DIR)/norm.c \
	$(SRC_DIR)/rope.c \
	$(SRC_DIR)/ffn.c \
	$(SRC_DIR)/kv_cache.c \
	$(SRC_DIR)/attention.c \
	$(SRC_DIR)/transformer_block.c

COMMON_OBJS := $(patsubst %.c,$(BUILD_DIR)/%.o,$(COMMON_SRCS))
SIM_OBJS := $(BUILD_DIR)/$(SRC_DIR)/main.o $(COMMON_OBJS)

.PHONY: all clean test
.SECONDARY:

all: $(SIM_BIN)

$(SIM_BIN): $(SIM_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD_DIR)/test_%: $(BUILD_DIR)/$(TEST_DIR)/test_%.o $(COMMON_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c -o $@ $<

test: $(TEST_BINS)
	@for test_bin in $(TEST_BINS); do \
		./$$test_bin; \
	done

clean:
	rm -rf $(BUILD_DIR)

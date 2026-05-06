CC ?= cc

BUILD_DIR := build
INCLUDE_DIR := include
SRC_DIR := src
TEST_DIR := tests

CFLAGS := -std=c11 -Wall -Wextra -Wpedantic -Werror -I$(INCLUDE_DIR)
LDFLAGS :=

SIM_BIN := $(BUILD_DIR)/att1-sim
TEST_BIN := $(BUILD_DIR)/test_smoke

COMMON_SRCS := $(SRC_DIR)/log.c
SIM_SRCS := $(SRC_DIR)/main.c $(COMMON_SRCS)
TEST_SRCS := $(TEST_DIR)/test_smoke.c $(COMMON_SRCS)

SIM_OBJS := $(patsubst %.c,$(BUILD_DIR)/%.o,$(SIM_SRCS))
TEST_OBJS := $(patsubst %.c,$(BUILD_DIR)/%.o,$(TEST_SRCS))

.PHONY: all clean test

all: $(SIM_BIN)

$(SIM_BIN): $(SIM_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

$(TEST_BIN): $(TEST_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c -o $@ $<

test: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -rf $(BUILD_DIR)

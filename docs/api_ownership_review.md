# ATT-1 API and Header Ownership Review (M141)

This document records the findings of the M141 review of all public ATT-1
headers for ownership, lifetime, mutability, shallow-copy hazards,
const-correctness, error/status consistency, public/private boundary clarity,
and thread-safety.

Narrow informational comments were added to affected headers (no behavior
changes, no API breaks). Larger refactoring items are listed in §8 for future
milestones.

---

## 1. Executive Summary

The ATT-1 public headers are in reasonable shape for a phase-1 simulator.
The main findings are:

| Finding | Severity | Disposition |
|---------|----------|-------------|
| Several structs with owned pointers lacked "must not be shallow-copied" comments | Medium | Fixed in M141 (comments added) |
| `att1_infer_set_trace` borrow lifetime undocumented | Medium | Fixed in M141 (comment added) |
| `att1_kv_cache`, `att1_kv_mmu` init functions return `int` rather than `att1_status_t` | Low | Deferred — would require callers to update |
| `att1_tensor_alloc_f32` returns `int` rather than `att1_status_t` | Low | Deferred |
| `att1_q8_matrix`, `att1_q4_matrix` alloc functions return `int` not `att1_status_t` | Low | Deferred |
| `att1_kv_mmu` is not opaque — internal pointer fields exposed | Low | Deferred — no external callers iterate internal page/session arrays |
| `att1_status_t` has two alias duplicates (`ATT1_ERR_INVALID`, `ATT1_ERR_NO_MEMORY`) | Low | Deferred — aliases are backward-compat guards |
| AIMU error codes (`att1_aimu_result`) are a parallel enum to `att1_status_t` | Info | By design; AIMU protocol codes are wire-format values |
| No thread-safety annotations or documentation | Info | Added to review doc (§6); no code change |

No behavior-changing fixes were made. All headers compile cleanly.
**781 PASS 0 FAIL (C) maintained.**

---

## 2. Header-by-Header Review

### 2.1 `att1_status.h`

**Purpose**: Shared error-code enum used by most APIs.

| Item | Finding |
|------|---------|
| `ATT1_ERR_INVALID` | Alias of `ATT1_ERR_INVALID_ARG` — kept for backward compat, but prefer the full name in new code |
| `ATT1_ERR_NO_MEMORY` | Alias of `ATT1_ERR_OOM` — same as above |
| Missing: `ATT1_ERR_DTYPE` | Some APIs that reject a dtype use `ATT1_ERR_UNSUPPORTED` or `ATT1_ERR_BAD_FORMAT` — inconsistently |
| Enum is `typedef att1_status` → `att1_status_t` | Clean; consistent naming |

**Recommended future fix**: Add a `ATT1_ERR_DTYPE` code for dtype-rejection paths to distinguish from format errors.

---

### 2.2 `att1_model.h`

**Purpose**: `.att1` model file loading and tensor access.

| Item | Finding |
|------|---------|
| `att1_model` owns `tensors[]` and `file_data` | ✓ documented in loader comment |
| `att1_model` must not be shallow-copied | Added comment in M141 |
| `att1_model_tensor.data` borrows into `file_data` | Valid only while parent `att1_model` lives; added lifetime comment in M141 |
| `att1_model_config` — no pointers | Safe to copy by value |
| `att1_model_info.name` borrows from model | Valid only while model lives |
| `att1_shard_meta`, `att1_tok_meta` embedded by value | Safe (both are value structs with no heap pointers) |
| `att1_model_find_tensor` returns `const att1_model_tensor *` | ✓ correct const annotation |
| `att1_model_load` takes `const char *` | ✓ correct |

**Lifecycle**:
```c
att1_model model;
att1_model_load(path, &model);  /* model owns file_data + tensors */
/* ... use model ... */
att1_model_free(&model);        /* required; do not shallow-copy model */
```

---

### 2.3 `att1_tensor.h`

**Purpose**: Owned float32 tensor allocation.

| Item | Finding |
|------|---------|
| `att1_tensor` owns `data` (float *) | Must not be shallow-copied; added comment in M141 |
| `att1_tensor_alloc_f32` returns `int` | Inconsistency: other alloc APIs return `att1_status_t`; deferred |
| `att1_tensor_free(NULL)` allowed | ✓ |
| `att1_tensor_desc` — no pointers | Safe to copy by value |
| `att1_tensor.data` is `float *` (not `const float *`) | Intentional — tensor is writable storage |

**Lifecycle**:
```c
att1_tensor t;
att1_tensor_alloc_f32(&t, rank, shape);  /* t owns data */
/* ... use t.data ... */
att1_tensor_free(&t);                    /* required */
```

---

### 2.4 `att1_backend.h`

**Purpose**: Backend dispatch table and lifecycle.

| Item | Finding |
|------|---------|
| `att1_backend_ops` is a vtable struct | Functions return `int` (0=success, -1=failure), not `att1_status_t` — deferred |
| `att1_backend_cpu_f32_create` etc. return `att1_status_t` | ✓ consistent |
| `att1_backend_cuda_available` returns `int` | Intentional boolean predicate — correct |
| `att1_backend_destroy` for cleanup | ✓ |
| `att1_infer_set_backend` takes ownership | Documented in `att1_infer.h` |
| `att1_backend.user_data` is `void *` | Backend implementors manage this lifetime |
| `att1_backend` struct is semi-opaque | `ops` and `user_data` are public fields — workable for phase-1 extensibility |

**High-risk**: After `att1_infer_set_backend(infer, backend)`, the caller
must not call `att1_backend_destroy(backend)` — infer now owns it.

---

### 2.5 `att1_infer.h`

**Purpose**: Single-tile inference context lifecycle and decode API.

| Item | Finding |
|------|---------|
| `att1_infer_t` is opaque | ✓ — best practice |
| `att1_infer_create` takes `const att1_model *` | ✓ correct borrow |
| `att1_infer_logits` returns `const float *` | ✓ correct const; pointer borrows into internal buffer |
| `att1_infer_logits` return pointer lifetime | Valid only until next decode call or `att1_infer_destroy`; added comment in M141 |
| `att1_infer_set_trace(infer, trace)` | **Borrows** trace — does NOT take ownership; caller must keep trace alive; added comment in M141 |
| `att1_infer_set_backend` | Takes ownership of backend; documented |
| `att1_infer_position`, `att1_infer_layer_kv_length` | Return `att1_status_t` through out-param — ✓ consistent |

**Lifetime rule for logits pointer**:
```c
const float *logits;
size_t n;
att1_infer_decode_token(infer, tok, &next);
logits = att1_infer_logits(infer, &n);
/* logits is valid here */
att1_infer_decode_token(infer, tok2, &next);
/* logits MAY be invalidated — do not use after next decode */
```

---

### 2.6 `att1_cluster_infer.h`

**Purpose**: Multi-tile cluster inference context.

| Item | Finding |
|------|---------|
| `att1_cluster_infer_t` is opaque | ✓ |
| `att1_cluster_infer_config` — no pointers | Safe to copy by value |
| `att1_cluster_tile_counters` — no pointers | Safe to copy by value |
| Cluster context borrows `att1_model *` | Caller must keep model alive while context lives |

---

### 2.7 `att1_quant.h`

**Purpose**: q8 and q4 quantization matrix types and primitives.

| Item | Finding |
|------|---------|
| `att1_q8_matrix` owns `values` and `scales` | Must not be shallow-copied; added comment in M141 |
| `att1_q4_matrix` owns `packed` and `scales` | Must not be shallow-copied; added comment in M141 |
| `att1_q8_matrix_alloc` returns `int` | Inconsistency with `att1_status_t` pattern; deferred |
| `att1_q4_matrix_alloc` returns `int` | Same; deferred |
| `att1_q4_group_scale`, `att1_q4_pack_group` etc. return `int` | These are low-level primitives; return code is -1/0; acceptable |
| `att1_quant_desc` — no pointers | Safe to copy |
| Wire-format constants (group size, flags) | Well documented with `#define` names |
| `att1_quantize_q4_per_group` allocates internally | Caller must free with `att1_q4_matrix_free` — documented |

---

### 2.8 `att1_shard.h`

**Purpose**: Layer-to-tile sharding plan.

| Item | Finding |
|------|---------|
| `att1_shard_plan` owns `tiles[]` and `layer_to_tile[]` | Must not be shallow-copied; added comment in M141 |
| `att1_layer_shard` — no pointers | Safe to copy |
| `att1_meta_plan` owns `entries[]` | Struct used internally; added comment |
| `att1_shard_plan_tile` returns pointer into plan | Valid only while plan lives |
| `att1_shard_plan_build` borrows `const att1_model *` | ✓ correct |

---

### 2.9 `att1_kv_cache.h`

**Purpose**: Simple contiguous KV cache (Milestone 2).

| Item | Finding |
|------|---------|
| `att1_kv_cache` owns `keys` and `values` | Must not be shallow-copied; added comment in M141 |
| `att1_kv_cache_init` returns `int` | Inconsistency; deferred |
| `att1_kv_cache_key` / `_value` return `const float *` | ✓ correct const; pointer lifetime added in M141 |
| Return pointer lifetime | Valid only while cache lives and position ≤ cache->length |

---

### 2.10 `att1_kv_mmu.h`

**Purpose**: Paged KV-cache MMU simulator.

| Item | Finding |
|------|---------|
| `att1_kv_mmu` is not opaque | Exposes `sessions *` and `pages *` — internal implementation details |
| `att1_kv_mmu_page` has owned `token_present`, `keys`, `values` | Internal struct with owned pointers; must not be copied by callers |
| `att1_kv_mmu_session` is visible | Fine for phase-1; no external callers iterate it |
| `att1_kv_mmu` must not be shallow-copied | Added comment in M141 |
| `att1_kv_mmu_init` returns `int` | Inconsistency; deferred |
| `att1_kv_mmu_lookup_page` returns `att1_kv_mmu_page_ref` | Value struct (no pointers); safe to return by out-param |
| `att1_kv_mmu_counters` — no pointers | Safe to copy |

**Recommended future fix**: Make `att1_kv_mmu` opaque in a later milestone.

---

### 2.11 `att1_tok_meta.h` and `att1_tok_ext.h`

**Purpose**: Tokenizer metadata section parsing.

| Item | Finding |
|------|---------|
| `att1_tok_meta` — no heap pointers | Safe to copy; inline `asset_hash[32]` array |
| `att1_tok_meta_parse` returns `att1_status_t` | ✓ consistent |
| `att1_tok_type_name` returns `const char *` | Points to static string literal — always valid |

---

### 2.12 `att1_shard_meta.h`

**Purpose**: Shard metadata section.

| Item | Finding |
|------|---------|
| Embedded by value in `att1_model` | Fine if it carries no heap pointers |
| Review confirms no heap pointers in `att1_shard_meta` | ✓ safe to embed by value |

---

### 2.13 AIMU headers (`att1_aimu_*.h`)

**Purpose**: AIMU command queue, MMIO, DMA, device, host, exec simulation.

| Item | Finding |
|------|---------|
| `att1_aimu_cmd` is a 64-byte wire-format struct | Safe to copy — wire record, no heap pointers |
| `att1_aimu_result` is a separate enum from `att1_status_t` | By design; AIMU protocol codes are hardware wire values |
| AIMU context types (cmdq, device, etc.) appear opaque | ✓ good practice |
| `ATT1_AIMU_CMDQ_MAGIC` / size check `att1_aimu_cmd_size_check` | ✓ static-assert pattern for wire struct size |
| `att1_aimu_userspace` | Userspace MMIO emulation; not a real driver |

---

## 3. High-Risk Structs / APIs

The following require caller care to avoid double-free or use-after-free:

| Struct / API | Risk | Rule |
|---|---|---|
| `att1_model` | Double-free on shallow copy | Never copy; always use `att1_model_free` |
| `att1_tensor` | Double-free on shallow copy | Never copy; use `att1_tensor_free` |
| `att1_q8_matrix` | Double-free on shallow copy | Never copy; use `att1_q8_matrix_free` |
| `att1_q4_matrix` | Double-free on shallow copy | Never copy; use `att1_q4_matrix_free` |
| `att1_kv_cache` | Double-free on shallow copy | Never copy; use `att1_kv_cache_free` |
| `att1_kv_mmu` | Double-free on shallow copy | Never copy; use `att1_kv_mmu_free` |
| `att1_shard_plan` | Double-free on shallow copy | Never copy; use `att1_shard_plan_free` |
| `att1_backend` after `att1_infer_set_backend` | Use-after-free if caller destroys | Infer owns backend after `set_backend` |
| `att1_infer_logits` return value | Use-after-free after next decode | Copy logits buffer if needed across decodes |
| `att1_infer_set_trace` borrow | Use-after-free if trace destroyed first | Caller keeps trace alive while infer runs |
| `att1_kv_cache_key` / `_value` return | Dangling pointer if cache freed | Use immediately; do not store across cache lifetime |

---

## 4. Const-Correctness Summary

| API | Const status |
|-----|-------------|
| `att1_model_load(const char *path, ...)` | ✓ |
| `att1_model_find_tensor(const att1_model *, ...)` | ✓ |
| `att1_infer_create(const att1_model *, ...)` | ✓ |
| `att1_infer_logits(const att1_infer_t *, ...)` | ✓ |
| `att1_kv_cache_key(const att1_kv_cache *, ...)` | ✓ |
| `att1_kv_mmu_get_counters(const att1_kv_mmu *, ...)` | ✓ |
| `att1_shard_plan_build(plan, const att1_model *, ...)` | ✓ |
| `att1_matmul_q8xf32(dst, const float *lhs, ..., const att1_q8_matrix *)` | ✓ |
| `att1_matmul_q4xf32(dst, const float *lhs, ..., const att1_q4_matrix *)` | ✓ |
| `att1_infer_set_trace(att1_infer_t *, att1_trace_t *)` | trace not `const` — intentional (infer calls mutable trace methods) |
| `att1_kv_mmu_lookup_page(att1_kv_mmu *, ...)` | Not `const` — updates hit/miss counters; acceptable |

Overall const-correctness is good. No missing `const` qualifiers found that
would be safe to add without touching implementation files.

---

## 5. Error / Status Consistency

| Pattern | Used by | Consistency |
|---------|---------|-------------|
| `att1_status_t` (enum, negative on error) | Most lifecycle/IO APIs | ✓ preferred |
| `int` returning 0/-1 | KV cache, KV MMU, tensor alloc, quant primitives | Inconsistent with `att1_status_t` |
| `att1_aimu_result` (wire enum) | AIMU command simulator | Intentional parallel system |

**Specific inconsistencies (deferred to future milestone)**:

- `att1_kv_cache_init` → `int`; should be `att1_status_t`
- `att1_kv_mmu_init` → `int`; should be `att1_status_t`
- `att1_tensor_alloc_f32` → `int`; should be `att1_status_t`
- `att1_q8_matrix_alloc`, `att1_q4_matrix_alloc` → `int`; should be `att1_status_t`
- `att1_q8_matrix_free`, `att1_q4_matrix_free`, `att1_kv_cache_free` → `void`; correct (free cannot fail)

These inconsistencies are low risk in practice because callers check `!= 0`
and the actual error semantics are the same (-1 vs negative `att1_status_t`).
However, moving to `att1_status_t` uniformly would allow callers to print
named error codes.

---

## 6. Thread-Safety and Reentrancy Notes

No ATT-1 public API is documented as thread-safe. The following rules apply:

| Component | Thread-safety |
|-----------|--------------|
| `att1_model` (after load) | Read-only access is safe from multiple threads; no locks needed for reads |
| `att1_infer_t` | **Not thread-safe** — single-threaded decode assumed |
| `att1_cluster_infer_t` | Uses pthreads internally; public API is not reentrant |
| `att1_kv_cache` | **Not thread-safe** — external synchronization required if shared |
| `att1_kv_mmu` | **Not thread-safe** — counters and page arrays mutated without locks |
| `att1_backend` ops | Depends on implementation; CPU backends are not thread-safe by default |
| `att1_cluster_tile_counters` | Not live-updated; safe to read after cluster decode completes |
| AIMU simulator | Ring-buffer state is single-process; no real PCIe concurrency |

**Recommendation**: Add `/* not thread-safe */` comments to `att1_kv_cache`,
`att1_kv_mmu`, and `att1_infer_t` in a future cleanup milestone.

---

## 7. ABI / Versioning Notes

| Struct | ABI risk |
|--------|----------|
| `att1_model_config` | Stable wire-facing fields; adding fields is a break |
| `att1_model_tensor` | Wire-facing; fixed layout |
| `att1_aimu_cmd` | **Hard wire format** — 64-byte struct with static-assert; must not change |
| `att1_tok_meta` | 96-byte wire format; must not change without version bump |
| `att1_kv_mmu` | Not opaque — any field reorder or add breaks callers |
| `att1_backend_ops` | Vtable — adding new ops is an ABI break for external implementors |
| `att1_shard_plan` | Not opaque — field layout exposed |

**Recommended rule**: Any struct that is a wire-format record (loaded from or
written to `.att1` files or protocol packets) must be treated as frozen.
Runtime-only structs (`att1_infer_t`, backends) should eventually be made
opaque.

---

## 8. Recommended Future Fixes

These items were NOT changed in M141. Each is a candidate for a future
targeted milestone.

| # | Item | Suggested milestone |
|---|------|-------------------|
| 1 | Migrate `att1_kv_cache_init`, `att1_kv_mmu_init`, `att1_tensor_alloc_f32`, `att1_q8_matrix_alloc`, `att1_q4_matrix_alloc` to return `att1_status_t` | M148 |
| 2 | Make `att1_kv_mmu` opaque (hide `sessions` and `pages` pointers) | M148 |
| 3 | Add `/* not thread-safe */` annotations to `att1_kv_cache`, `att1_kv_mmu`, `att1_infer_t` | M148 |
| 4 | Add `ATT1_ERR_DTYPE` status code and use it consistently for dtype-rejection paths | M148 |
| 5 | Remove deprecated aliases `ATT1_ERR_INVALID` and `ATT1_ERR_NO_MEMORY` (after audit of callers) | M148 |
| 6 | Make `att1_shard_plan` opaque | M148 |
| 7 | `att1_backend_ops` vtable: add new ops only via capability query to avoid ABI breaks | M146+ |

---

## 9. Changes Made in M141

The following informational comments were added to public headers (no behavior
changes, no API breaks, no new symbols):

| Header | Change |
|--------|--------|
| `att1_model.h` | Added `/* must not be shallow-copied */` to `att1_model`; added lifetime note to `att1_model_tensor.data` |
| `att1_tensor.h` | Added `/* must not be shallow-copied; owns data */` to `att1_tensor` |
| `att1_quant.h` | Added `/* must not be shallow-copied */` to `att1_q8_matrix` and `att1_q4_matrix` |
| `att1_kv_cache.h` | Added `/* must not be shallow-copied; owns keys and values */` to `att1_kv_cache`; added lifetime note to `att1_kv_cache_key` / `_value` return |
| `att1_kv_mmu.h` | Added `/* must not be shallow-copied */` to `att1_kv_mmu`; added note on `att1_kv_mmu_page` internal owned pointers |
| `att1_shard.h` | Added `/* must not be shallow-copied */` to `att1_shard_plan` |
| `att1_infer.h` | Clarified that `att1_infer_set_trace` borrows (does not take ownership); added logits pointer lifetime note |

#include "att1_aimu_conformance.h"

#include "att1_aimu_device.h"

#include <stdlib.h>
#include <string.h>

typedef struct att1_aimu_conformance_inproc {
    att1_aimu_conformance_config config;
    att1_aimu_device *device;
    att1_aimu_cmdq *cmdq;
    att1_aimu_dma *dma;
    att1_aimu_trace *trace;
    att1_aimu_mmio *mmio;
    att1_fabric fabric;
    att1_kv_mmu *kv_mmu;
} att1_aimu_conformance_inproc;

#define ATT1_AIMU_CONF_DEFAULT_TILE_COUNT            4u
#define ATT1_AIMU_CONF_DEFAULT_TILE_MEMORY_BYTES     (UINT64_C(1) << 30)
#define ATT1_AIMU_CONF_DEFAULT_TILE_KV_BYTES         (UINT64_C(256) << 20)
#define ATT1_AIMU_CONF_DEFAULT_RING_DEPTH            64u
#define ATT1_AIMU_CONF_DEFAULT_FABRIC_QUEUE_CAP      3u
#define ATT1_AIMU_CONF_DEFAULT_FABRIC_MAX_PAYLOAD    256u
#define ATT1_AIMU_CONF_DEFAULT_DMA_HOST_BASE         UINT64_C(0x0000000000010000)
#define ATT1_AIMU_CONF_DEFAULT_DMA_HOST_SIZE         UINT64_C(0x0000000010000000)
#define ATT1_AIMU_CONF_DEFAULT_DMA_DEVICE_BASE       UINT64_C(0x0000000080000000)
#define ATT1_AIMU_CONF_DEFAULT_DMA_DEVICE_SIZE       UINT64_C(0x0000000040000000)
#define ATT1_AIMU_CONF_DEFAULT_KV_MAX_SESSIONS       16u
#define ATT1_AIMU_CONF_DEFAULT_KV_MAX_PAGES          256u
#define ATT1_AIMU_CONF_DEFAULT_KV_NUM_LAYERS         4u
#define ATT1_AIMU_CONF_DEFAULT_KV_NUM_HEADS          4u
#define ATT1_AIMU_CONF_DEFAULT_KV_HEAD_DIM           16u
#define ATT1_AIMU_CONF_DEFAULT_KV_PAGE_TOKENS        16u
#define ATT1_AIMU_CONF_DEFAULT_KV_MAX_POSITIONS      256u

static att1_status_t endpoint_invalid(void)
{
    return ATT1_ERR_INVALID_ARG;
}

static att1_aimu_conformance_inproc *endpoint_ctx(void *ctx)
{
    return (att1_aimu_conformance_inproc *)ctx;
}

void att1_aimu_conformance_default_config(att1_aimu_conformance_config *out)
{
    if (out == NULL) {
        return;
    }

    memset(out, 0, sizeof(*out));
    out->tile_count = ATT1_AIMU_CONF_DEFAULT_TILE_COUNT;
    out->tile_memory_bytes = ATT1_AIMU_CONF_DEFAULT_TILE_MEMORY_BYTES;
    out->tile_kv_bytes = ATT1_AIMU_CONF_DEFAULT_TILE_KV_BYTES;
    out->cmd_ring_depth = ATT1_AIMU_CONF_DEFAULT_RING_DEPTH;
    out->comp_ring_depth = ATT1_AIMU_CONF_DEFAULT_RING_DEPTH;
    out->fabric_queue_capacity = ATT1_AIMU_CONF_DEFAULT_FABRIC_QUEUE_CAP;
    out->fabric_max_payload_bytes = ATT1_AIMU_CONF_DEFAULT_FABRIC_MAX_PAYLOAD;
    out->dma_host_base = ATT1_AIMU_CONF_DEFAULT_DMA_HOST_BASE;
    out->dma_host_size = ATT1_AIMU_CONF_DEFAULT_DMA_HOST_SIZE;
    out->dma_device_base = ATT1_AIMU_CONF_DEFAULT_DMA_DEVICE_BASE;
    out->dma_device_size = ATT1_AIMU_CONF_DEFAULT_DMA_DEVICE_SIZE;
    out->kv_max_sessions = ATT1_AIMU_CONF_DEFAULT_KV_MAX_SESSIONS;
    out->kv_max_pages = ATT1_AIMU_CONF_DEFAULT_KV_MAX_PAGES;
    out->kv_num_layers = ATT1_AIMU_CONF_DEFAULT_KV_NUM_LAYERS;
    out->kv_num_heads = ATT1_AIMU_CONF_DEFAULT_KV_NUM_HEADS;
    out->kv_head_dim = ATT1_AIMU_CONF_DEFAULT_KV_HEAD_DIM;
    out->kv_page_tokens = ATT1_AIMU_CONF_DEFAULT_KV_PAGE_TOKENS;
    out->kv_max_positions = ATT1_AIMU_CONF_DEFAULT_KV_MAX_POSITIONS;
}

static void conformance_fill_config(att1_aimu_conformance_config *cfg,
                                    const att1_aimu_conformance_config *in)
{
    att1_aimu_conformance_default_config(cfg);
    if (in == NULL) {
        return;
    }

    if (in->tile_count != 0u) {
        cfg->tile_count = in->tile_count;
    }
    if (in->tile_memory_bytes != 0u) {
        cfg->tile_memory_bytes = in->tile_memory_bytes;
    }
    if (in->tile_kv_bytes != 0u) {
        cfg->tile_kv_bytes = in->tile_kv_bytes;
    }
    if (in->cmd_ring_depth != 0u) {
        cfg->cmd_ring_depth = in->cmd_ring_depth;
    }
    if (in->comp_ring_depth != 0u) {
        cfg->comp_ring_depth = in->comp_ring_depth;
    }
    if (in->fabric_queue_capacity != 0u) {
        cfg->fabric_queue_capacity = in->fabric_queue_capacity;
    }
    if (in->fabric_max_payload_bytes != 0u) {
        cfg->fabric_max_payload_bytes = in->fabric_max_payload_bytes;
    }
    if (in->dma_host_base != 0u) {
        cfg->dma_host_base = in->dma_host_base;
    }
    if (in->dma_host_size != 0u) {
        cfg->dma_host_size = in->dma_host_size;
    }
    if (in->dma_device_base != 0u) {
        cfg->dma_device_base = in->dma_device_base;
    }
    if (in->dma_device_size != 0u) {
        cfg->dma_device_size = in->dma_device_size;
    }
    if (in->kv_max_sessions != 0u) {
        cfg->kv_max_sessions = in->kv_max_sessions;
    }
    if (in->kv_max_pages != 0u) {
        cfg->kv_max_pages = in->kv_max_pages;
    }
    if (in->kv_num_layers != 0u) {
        cfg->kv_num_layers = in->kv_num_layers;
    }
    if (in->kv_num_heads != 0u) {
        cfg->kv_num_heads = in->kv_num_heads;
    }
    if (in->kv_head_dim != 0u) {
        cfg->kv_head_dim = in->kv_head_dim;
    }
    if (in->kv_page_tokens != 0u) {
        cfg->kv_page_tokens = in->kv_page_tokens;
    }
    if (in->kv_max_positions != 0u) {
        cfg->kv_max_positions = in->kv_max_positions;
    }
}

static void inproc_destroy(void *ctx)
{
    att1_aimu_conformance_inproc *ep = endpoint_ctx(ctx);
    if (ep == NULL) {
        return;
    }

    att1_fabric_destroy(&ep->fabric);
    att1_aimu_mmio_destroy(ep->mmio);
    att1_aimu_trace_destroy(ep->trace);
    att1_aimu_dma_destroy(ep->dma);
    att1_aimu_cmdq_destroy(ep->cmdq);
    att1_aimu_device_destroy(ep->device);
    att1_kv_mmu_destroy(ep->kv_mmu);
    free(ep);
}

static att1_status_t inproc_sync_mmio(void *ctx)
{
    att1_aimu_conformance_inproc *ep = endpoint_ctx(ctx);
    if (ep == NULL) {
        return endpoint_invalid();
    }
    return att1_aimu_mmio_sync(ep->mmio);
}

static att1_status_t inproc_snapshot_counters(void *ctx)
{
    att1_aimu_conformance_inproc *ep = endpoint_ctx(ctx);
    att1_status_t status;

    if (ep == NULL) {
        return endpoint_invalid();
    }

    status = att1_aimu_trace_snapshot_all(ep->trace, ep->cmdq, ep->device, ep->dma);
    if (status != ATT1_OK) {
        return status;
    }
    return att1_aimu_mmio_sync(ep->mmio);
}

static att1_status_t inproc_mmio_read32(void *ctx, uint32_t offset, uint32_t *out)
{
    att1_aimu_conformance_inproc *ep = endpoint_ctx(ctx);
    if (ep == NULL) {
        return endpoint_invalid();
    }
    return att1_aimu_mmio_read32(ep->mmio, offset, out);
}

static att1_status_t inproc_mmio_write32(void *ctx, uint32_t offset, uint32_t value)
{
    att1_aimu_conformance_inproc *ep = endpoint_ctx(ctx);
    if (ep == NULL) {
        return endpoint_invalid();
    }
    return att1_aimu_mmio_write32(ep->mmio, offset, value);
}

static att1_status_t inproc_mmio_read64(void *ctx, uint32_t offset, uint64_t *out)
{
    att1_aimu_conformance_inproc *ep = endpoint_ctx(ctx);
    if (ep == NULL) {
        return endpoint_invalid();
    }
    return att1_aimu_mmio_read64(ep->mmio, offset, out);
}

static att1_status_t inproc_mmio_write64(void *ctx, uint32_t offset, uint64_t value)
{
    att1_aimu_conformance_inproc *ep = endpoint_ctx(ctx);
    if (ep == NULL) {
        return endpoint_invalid();
    }
    return att1_aimu_mmio_write64(ep->mmio, offset, value);
}

static att1_status_t inproc_cmd_submit(void *ctx, att1_aimu_cmd *cmd)
{
    att1_aimu_conformance_inproc *ep = endpoint_ctx(ctx);
    if (ep == NULL) {
        return endpoint_invalid();
    }
    return att1_aimu_cmdq_submit(ep->cmdq, cmd);
}

static att1_status_t inproc_cmd_dispatch_one(void *ctx)
{
    att1_aimu_conformance_inproc *ep = endpoint_ctx(ctx);
    if (ep == NULL) {
        return endpoint_invalid();
    }
    return att1_aimu_cmdq_dispatch_one(ep->cmdq);
}

static att1_status_t inproc_cmd_dispatch_all(void *ctx)
{
    att1_aimu_conformance_inproc *ep = endpoint_ctx(ctx);
    if (ep == NULL) {
        return endpoint_invalid();
    }
    return att1_aimu_cmdq_dispatch_all(ep->cmdq);
}

static att1_status_t inproc_cmd_poll_completion(void *ctx, att1_aimu_completion *out)
{
    att1_aimu_conformance_inproc *ep = endpoint_ctx(ctx);
    if (ep == NULL) {
        return endpoint_invalid();
    }
    return att1_aimu_cmdq_poll_completion(ep->cmdq, out);
}

static att1_status_t inproc_cmd_get_counters(void *ctx, att1_aimu_cmdq_counters *out)
{
    att1_aimu_conformance_inproc *ep = endpoint_ctx(ctx);
    if (ep == NULL) {
        return endpoint_invalid();
    }
    return att1_aimu_cmdq_get_counters(ep->cmdq, out);
}

static att1_status_t inproc_dma_validate(void *ctx, const att1_aimu_dma_desc *desc)
{
    att1_aimu_conformance_inproc *ep = endpoint_ctx(ctx);
    if (ep == NULL) {
        return endpoint_invalid();
    }
    return att1_aimu_dma_validate(ep->dma, desc);
}

static att1_status_t inproc_dma_submit(void *ctx, const att1_aimu_dma_desc *desc)
{
    att1_aimu_conformance_inproc *ep = endpoint_ctx(ctx);
    if (ep == NULL) {
        return endpoint_invalid();
    }
    return att1_aimu_dma_submit(ep->dma, desc);
}

static att1_status_t inproc_dma_get_counters(void *ctx, att1_aimu_dma_counters *out)
{
    att1_aimu_conformance_inproc *ep = endpoint_ctx(ctx);
    if (ep == NULL) {
        return endpoint_invalid();
    }
    return att1_aimu_dma_get_counters(ep->dma, out);
}

static att1_status_t inproc_fabric_send(void *ctx,
                                        uint32_t source_tile,
                                        uint32_t target_tile,
                                        att1_packet_type type,
                                        const void *payload,
                                        size_t payload_bytes,
                                        uint64_t tag)
{
    att1_aimu_conformance_inproc *ep = endpoint_ctx(ctx);
    if (ep == NULL) {
        return endpoint_invalid();
    }
    return att1_fabric_send(&ep->fabric,
                            source_tile,
                            target_tile,
                            type,
                            payload,
                            payload_bytes,
                            tag);
}

static att1_status_t inproc_fabric_broadcast(void *ctx,
                                             uint32_t source_tile,
                                             const uint32_t *group_tiles,
                                             size_t group_count,
                                             att1_packet_type type,
                                             const void *payload,
                                             size_t payload_bytes,
                                             uint64_t tag)
{
    att1_aimu_conformance_inproc *ep = endpoint_ctx(ctx);
    if (ep == NULL) {
        return endpoint_invalid();
    }
    return att1_fabric_broadcast(&ep->fabric,
                                 source_tile,
                                 group_tiles,
                                 group_count,
                                 type,
                                 payload,
                                 payload_bytes,
                                 tag);
}

static att1_status_t inproc_fabric_receive(void *ctx,
                                           uint32_t tile_id,
                                           att1_fabric_packet *out_packet,
                                           void *out_payload,
                                           size_t out_payload_capacity,
                                           size_t *out_payload_bytes)
{
    att1_aimu_conformance_inproc *ep = endpoint_ctx(ctx);
    if (ep == NULL) {
        return endpoint_invalid();
    }
    return att1_fabric_receive(&ep->fabric,
                               tile_id,
                               out_packet,
                               out_payload,
                               out_payload_capacity,
                               out_payload_bytes);
}

static att1_status_t inproc_fabric_barrier_arrive(void *ctx,
                                                  uint32_t tile_id,
                                                  const uint32_t *participants,
                                                  size_t participant_count,
                                                  int *out_complete)
{
    att1_aimu_conformance_inproc *ep = endpoint_ctx(ctx);
    if (ep == NULL) {
        return endpoint_invalid();
    }
    return att1_fabric_barrier_arrive(&ep->fabric,
                                      tile_id,
                                      participants,
                                      participant_count,
                                      out_complete);
}

static att1_status_t inproc_fabric_get_counters(void *ctx, att1_fabric_counters *out)
{
    att1_aimu_conformance_inproc *ep = endpoint_ctx(ctx);
    if (ep == NULL || out == NULL) {
        return endpoint_invalid();
    }
    att1_fabric_get_counters(&ep->fabric, out);
    return ATT1_OK;
}

static att1_status_t inproc_trace_get_snapshot(void *ctx, att1_aimu_trace_snapshot *out)
{
    att1_aimu_conformance_inproc *ep = endpoint_ctx(ctx);
    if (ep == NULL) {
        return endpoint_invalid();
    }
    return att1_aimu_trace_get_snapshot(ep->trace, out);
}

static att1_status_t inproc_kv_create_session(void *ctx, uint64_t session_id)
{
    att1_aimu_conformance_inproc *ep = endpoint_ctx(ctx);
    if (ep == NULL) {
        return endpoint_invalid();
    }
    return att1_kv_mmu_create_session(ep->kv_mmu, session_id);
}

static att1_status_t inproc_kv_destroy_session(void *ctx, uint64_t session_id)
{
    att1_aimu_conformance_inproc *ep = endpoint_ctx(ctx);
    if (ep == NULL) {
        return endpoint_invalid();
    }
    return att1_kv_mmu_destroy_session(ep->kv_mmu, session_id);
}

static att1_status_t inproc_kv_append(void *ctx,
                                      uint64_t session_id,
                                      size_t layer_id,
                                      size_t position,
                                      const float *key,
                                      size_t key_count,
                                      const float *value,
                                      size_t value_count)
{
    att1_aimu_conformance_inproc *ep = endpoint_ctx(ctx);
    (void)key_count;
    (void)value_count;
    if (ep == NULL) {
        return endpoint_invalid();
    }
    return att1_kv_mmu_append(ep->kv_mmu, session_id, layer_id, position, key, value);
}

static att1_status_t inproc_kv_read(void *ctx,
                                    uint64_t session_id,
                                    size_t layer_id,
                                    size_t head_id,
                                    size_t position,
                                    float *out_key,
                                    size_t key_count,
                                    float *out_value,
                                    size_t value_count)
{
    att1_aimu_conformance_inproc *ep = endpoint_ctx(ctx);
    (void)key_count;
    (void)value_count;
    if (ep == NULL) {
        return endpoint_invalid();
    }
    return att1_kv_mmu_read(ep->kv_mmu,
                            session_id,
                            layer_id,
                            head_id,
                            position,
                            out_key,
                            out_value);
}

static att1_status_t inproc_kv_copy_range(void *ctx,
                                          uint64_t session_id,
                                          size_t layer_id,
                                          size_t head_id,
                                          size_t start_position,
                                          size_t position_count,
                                          float *out_keys,
                                          size_t keys_count,
                                          float *out_values,
                                          size_t values_count)
{
    att1_aimu_conformance_inproc *ep = endpoint_ctx(ctx);
    (void)keys_count;
    (void)values_count;
    if (ep == NULL) {
        return endpoint_invalid();
    }
    return att1_kv_mmu_copy_range(ep->kv_mmu,
                                 session_id,
                                 layer_id,
                                 head_id,
                                 start_position,
                                 position_count,
                                 out_keys,
                                 out_values);
}

static att1_status_t inproc_kv_get_counters(void *ctx, att1_kv_mmu_counters *out)
{
    att1_aimu_conformance_inproc *ep = endpoint_ctx(ctx);
    if (ep == NULL || out == NULL) {
        return endpoint_invalid();
    }
    att1_kv_mmu_get_counters(ep->kv_mmu, out);
    return ATT1_OK;
}

static const att1_aimu_conformance_ops g_inproc_ops = {
    "aimu-inproc-sim",
    inproc_destroy,
    inproc_sync_mmio,
    inproc_snapshot_counters,
    inproc_mmio_read32,
    inproc_mmio_write32,
    inproc_mmio_read64,
    inproc_mmio_write64,
    inproc_cmd_submit,
    inproc_cmd_dispatch_one,
    inproc_cmd_dispatch_all,
    inproc_cmd_poll_completion,
    inproc_cmd_get_counters,
    inproc_dma_validate,
    inproc_dma_submit,
    inproc_dma_get_counters,
    inproc_fabric_send,
    inproc_fabric_broadcast,
    inproc_fabric_receive,
    inproc_fabric_barrier_arrive,
    inproc_fabric_get_counters,
    inproc_trace_get_snapshot,
    inproc_kv_create_session,
    inproc_kv_destroy_session,
    inproc_kv_append,
    inproc_kv_read,
    inproc_kv_copy_range,
    inproc_kv_get_counters
};

att1_status_t att1_aimu_conformance_inproc_create(
        const att1_aimu_conformance_config *config,
        att1_aimu_conformance_endpoint **out_endpoint)
{
    att1_aimu_conformance_inproc *ctx = NULL;
    att1_aimu_conformance_endpoint *endpoint = NULL;
    att1_aimu_conformance_config cfg;
    att1_aimu_device_config dev_cfg;
    att1_aimu_cmdq_config cmdq_cfg;
    att1_fabric_bus_config fabric_cfg;
    att1_kv_mmu_config kv_cfg;
    att1_status_t status;

    if (out_endpoint == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }
    *out_endpoint = NULL;

    conformance_fill_config(&cfg, config);

    endpoint = (att1_aimu_conformance_endpoint *)calloc(1u, sizeof(*endpoint));
    ctx = (att1_aimu_conformance_inproc *)calloc(1u, sizeof(*ctx));
    if (endpoint == NULL || ctx == NULL) {
        free(endpoint);
        free(ctx);
        return ATT1_ERR_OOM;
    }
    ctx->config = cfg;

    memset(&dev_cfg, 0, sizeof(dev_cfg));
    dev_cfg.tile_count = cfg.tile_count;
    dev_cfg.tile_memory_bytes = cfg.tile_memory_bytes;
    dev_cfg.tile_kv_bytes = cfg.tile_kv_bytes;
    status = att1_aimu_device_create(&dev_cfg, &ctx->device);
    if (status != ATT1_OK) {
        inproc_destroy(ctx);
        free(endpoint);
        return status;
    }

    memset(&cmdq_cfg, 0, sizeof(cmdq_cfg));
    cmdq_cfg.tile_count = cfg.tile_count;
    cmdq_cfg.cmd_ring_depth = cfg.cmd_ring_depth;
    cmdq_cfg.comp_ring_depth = cfg.comp_ring_depth;
    status = att1_aimu_cmdq_create(&cmdq_cfg, &ctx->cmdq);
    if (status != ATT1_OK) {
        inproc_destroy(ctx);
        free(endpoint);
        return status;
    }

    status = att1_aimu_device_attach_cmdq(ctx->device, ctx->cmdq);
    if (status != ATT1_OK) {
        inproc_destroy(ctx);
        free(endpoint);
        return status;
    }

    status = att1_aimu_dma_create(&ctx->dma);
    if (status != ATT1_OK) {
        inproc_destroy(ctx);
        free(endpoint);
        return status;
    }
    status = att1_aimu_dma_register_host_region(ctx->dma,
                                                cfg.dma_host_base,
                                                cfg.dma_host_size);
    if (status != ATT1_OK) {
        inproc_destroy(ctx);
        free(endpoint);
        return status;
    }
    status = att1_aimu_dma_register_device_region(ctx->dma,
                                                  cfg.dma_device_base,
                                                  cfg.dma_device_size);
    if (status != ATT1_OK) {
        inproc_destroy(ctx);
        free(endpoint);
        return status;
    }

    status = att1_aimu_trace_create(&ctx->trace);
    if (status != ATT1_OK) {
        inproc_destroy(ctx);
        free(endpoint);
        return status;
    }

    status = att1_aimu_mmio_create(&ctx->mmio);
    if (status != ATT1_OK) {
        inproc_destroy(ctx);
        free(endpoint);
        return status;
    }
    status = att1_aimu_mmio_attach_device(ctx->mmio, ctx->device);
    if (status != ATT1_OK) {
        inproc_destroy(ctx);
        free(endpoint);
        return status;
    }
    status = att1_aimu_mmio_attach_cmdq(ctx->mmio, ctx->cmdq);
    if (status != ATT1_OK) {
        inproc_destroy(ctx);
        free(endpoint);
        return status;
    }
    status = att1_aimu_mmio_attach_trace(ctx->mmio, ctx->trace);
    if (status != ATT1_OK) {
        inproc_destroy(ctx);
        free(endpoint);
        return status;
    }

    memset(&fabric_cfg, 0, sizeof(fabric_cfg));
    fabric_cfg.tile_count = cfg.tile_count;
    fabric_cfg.queue_capacity = cfg.fabric_queue_capacity;
    fabric_cfg.max_payload_bytes = cfg.fabric_max_payload_bytes;
    status = att1_fabric_create(&ctx->fabric, &fabric_cfg);
    if (status != ATT1_OK) {
        inproc_destroy(ctx);
        free(endpoint);
        return status;
    }

    memset(&kv_cfg, 0, sizeof(kv_cfg));
    kv_cfg.max_sessions = cfg.kv_max_sessions;
    kv_cfg.max_pages = cfg.kv_max_pages;
    kv_cfg.num_layers = cfg.kv_num_layers;
    kv_cfg.num_heads = cfg.kv_num_heads;
    kv_cfg.head_dim = cfg.kv_head_dim;
    kv_cfg.page_tokens = cfg.kv_page_tokens;
    kv_cfg.max_positions = cfg.kv_max_positions;
    status = att1_kv_mmu_create(&kv_cfg, &ctx->kv_mmu);
    if (status != ATT1_OK) {
        inproc_destroy(ctx);
        free(endpoint);
        return status;
    }

    status = att1_aimu_mmio_sync(ctx->mmio);
    if (status != ATT1_OK) {
        inproc_destroy(ctx);
        free(endpoint);
        return status;
    }

    endpoint->ops = &g_inproc_ops;
    endpoint->ctx = ctx;
    *out_endpoint = endpoint;
    return ATT1_OK;
}

static att1_status_t endpoint_validate(const att1_aimu_conformance_endpoint *endpoint)
{
    if (endpoint == NULL || endpoint->ops == NULL || endpoint->ctx == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }
    return ATT1_OK;
}

void att1_aimu_conformance_endpoint_destroy(att1_aimu_conformance_endpoint *endpoint)
{
    if (endpoint == NULL) {
        return;
    }
    if (endpoint->ops != NULL && endpoint->ops->destroy != NULL) {
        endpoint->ops->destroy(endpoint->ctx);
    }
    free(endpoint);
}

#define ATT1_CONF_VALIDATE_OP(EP, FIELD) \
    do { \
        if (endpoint_validate((EP)) != ATT1_OK || (EP)->ops->FIELD == NULL) { \
            return ATT1_ERR_INVALID_ARG; \
        } \
    } while (0)

att1_status_t att1_aimu_conformance_sync_mmio(att1_aimu_conformance_endpoint *endpoint)
{
    ATT1_CONF_VALIDATE_OP(endpoint, sync_mmio);
    return endpoint->ops->sync_mmio(endpoint->ctx);
}

att1_status_t att1_aimu_conformance_snapshot_counters(att1_aimu_conformance_endpoint *endpoint)
{
    ATT1_CONF_VALIDATE_OP(endpoint, snapshot_counters);
    return endpoint->ops->snapshot_counters(endpoint->ctx);
}

att1_status_t att1_aimu_conformance_mmio_read32(att1_aimu_conformance_endpoint *endpoint,
                                                uint32_t offset,
                                                uint32_t *out)
{
    ATT1_CONF_VALIDATE_OP(endpoint, mmio_read32);
    return endpoint->ops->mmio_read32(endpoint->ctx, offset, out);
}

att1_status_t att1_aimu_conformance_mmio_write32(att1_aimu_conformance_endpoint *endpoint,
                                                 uint32_t offset,
                                                 uint32_t value)
{
    ATT1_CONF_VALIDATE_OP(endpoint, mmio_write32);
    return endpoint->ops->mmio_write32(endpoint->ctx, offset, value);
}

att1_status_t att1_aimu_conformance_mmio_read64(att1_aimu_conformance_endpoint *endpoint,
                                                uint32_t offset,
                                                uint64_t *out)
{
    ATT1_CONF_VALIDATE_OP(endpoint, mmio_read64);
    return endpoint->ops->mmio_read64(endpoint->ctx, offset, out);
}

att1_status_t att1_aimu_conformance_mmio_write64(att1_aimu_conformance_endpoint *endpoint,
                                                 uint32_t offset,
                                                 uint64_t value)
{
    ATT1_CONF_VALIDATE_OP(endpoint, mmio_write64);
    return endpoint->ops->mmio_write64(endpoint->ctx, offset, value);
}

att1_status_t att1_aimu_conformance_cmd_submit(att1_aimu_conformance_endpoint *endpoint,
                                               att1_aimu_cmd *cmd)
{
    ATT1_CONF_VALIDATE_OP(endpoint, cmd_submit);
    return endpoint->ops->cmd_submit(endpoint->ctx, cmd);
}

att1_status_t att1_aimu_conformance_cmd_dispatch_one(att1_aimu_conformance_endpoint *endpoint)
{
    ATT1_CONF_VALIDATE_OP(endpoint, cmd_dispatch_one);
    return endpoint->ops->cmd_dispatch_one(endpoint->ctx);
}

att1_status_t att1_aimu_conformance_cmd_dispatch_all(att1_aimu_conformance_endpoint *endpoint)
{
    ATT1_CONF_VALIDATE_OP(endpoint, cmd_dispatch_all);
    return endpoint->ops->cmd_dispatch_all(endpoint->ctx);
}

att1_status_t att1_aimu_conformance_cmd_poll_completion(att1_aimu_conformance_endpoint *endpoint,
                                                        att1_aimu_completion *out)
{
    ATT1_CONF_VALIDATE_OP(endpoint, cmd_poll_completion);
    return endpoint->ops->cmd_poll_completion(endpoint->ctx, out);
}

att1_status_t att1_aimu_conformance_cmd_get_counters(att1_aimu_conformance_endpoint *endpoint,
                                                     att1_aimu_cmdq_counters *out)
{
    ATT1_CONF_VALIDATE_OP(endpoint, cmd_get_counters);
    return endpoint->ops->cmd_get_counters(endpoint->ctx, out);
}

att1_status_t att1_aimu_conformance_dma_validate(att1_aimu_conformance_endpoint *endpoint,
                                                 const att1_aimu_dma_desc *desc)
{
    ATT1_CONF_VALIDATE_OP(endpoint, dma_validate);
    return endpoint->ops->dma_validate(endpoint->ctx, desc);
}

att1_status_t att1_aimu_conformance_dma_submit(att1_aimu_conformance_endpoint *endpoint,
                                               const att1_aimu_dma_desc *desc)
{
    ATT1_CONF_VALIDATE_OP(endpoint, dma_submit);
    return endpoint->ops->dma_submit(endpoint->ctx, desc);
}

att1_status_t att1_aimu_conformance_dma_get_counters(att1_aimu_conformance_endpoint *endpoint,
                                                     att1_aimu_dma_counters *out)
{
    ATT1_CONF_VALIDATE_OP(endpoint, dma_get_counters);
    return endpoint->ops->dma_get_counters(endpoint->ctx, out);
}

att1_status_t att1_aimu_conformance_fabric_send(att1_aimu_conformance_endpoint *endpoint,
                                                uint32_t source_tile,
                                                uint32_t target_tile,
                                                att1_packet_type type,
                                                const void *payload,
                                                size_t payload_bytes,
                                                uint64_t tag)
{
    ATT1_CONF_VALIDATE_OP(endpoint, fabric_send);
    return endpoint->ops->fabric_send(endpoint->ctx,
                                      source_tile,
                                      target_tile,
                                      type,
                                      payload,
                                      payload_bytes,
                                      tag);
}

att1_status_t att1_aimu_conformance_fabric_broadcast(att1_aimu_conformance_endpoint *endpoint,
                                                     uint32_t source_tile,
                                                     const uint32_t *group_tiles,
                                                     size_t group_count,
                                                     att1_packet_type type,
                                                     const void *payload,
                                                     size_t payload_bytes,
                                                     uint64_t tag)
{
    ATT1_CONF_VALIDATE_OP(endpoint, fabric_broadcast);
    return endpoint->ops->fabric_broadcast(endpoint->ctx,
                                           source_tile,
                                           group_tiles,
                                           group_count,
                                           type,
                                           payload,
                                           payload_bytes,
                                           tag);
}

att1_status_t att1_aimu_conformance_fabric_receive(att1_aimu_conformance_endpoint *endpoint,
                                                   uint32_t tile_id,
                                                   att1_fabric_packet *out_packet,
                                                   void *out_payload,
                                                   size_t out_payload_capacity,
                                                   size_t *out_payload_bytes)
{
    ATT1_CONF_VALIDATE_OP(endpoint, fabric_receive);
    return endpoint->ops->fabric_receive(endpoint->ctx,
                                         tile_id,
                                         out_packet,
                                         out_payload,
                                         out_payload_capacity,
                                         out_payload_bytes);
}

att1_status_t att1_aimu_conformance_fabric_barrier_arrive(att1_aimu_conformance_endpoint *endpoint,
                                                          uint32_t tile_id,
                                                          const uint32_t *participants,
                                                          size_t participant_count,
                                                          int *out_complete)
{
    ATT1_CONF_VALIDATE_OP(endpoint, fabric_barrier_arrive);
    return endpoint->ops->fabric_barrier_arrive(endpoint->ctx,
                                                tile_id,
                                                participants,
                                                participant_count,
                                                out_complete);
}

att1_status_t att1_aimu_conformance_fabric_get_counters(att1_aimu_conformance_endpoint *endpoint,
                                                        att1_fabric_counters *out)
{
    ATT1_CONF_VALIDATE_OP(endpoint, fabric_get_counters);
    return endpoint->ops->fabric_get_counters(endpoint->ctx, out);
}

att1_status_t att1_aimu_conformance_trace_get_snapshot(att1_aimu_conformance_endpoint *endpoint,
                                                       att1_aimu_trace_snapshot *out)
{
    ATT1_CONF_VALIDATE_OP(endpoint, trace_get_snapshot);
    return endpoint->ops->trace_get_snapshot(endpoint->ctx, out);
}

att1_status_t att1_aimu_conformance_kv_create_session(att1_aimu_conformance_endpoint *endpoint,
                                                      uint64_t session_id)
{
    ATT1_CONF_VALIDATE_OP(endpoint, kv_create_session);
    return endpoint->ops->kv_create_session(endpoint->ctx, session_id);
}

att1_status_t att1_aimu_conformance_kv_destroy_session(att1_aimu_conformance_endpoint *endpoint,
                                                       uint64_t session_id)
{
    ATT1_CONF_VALIDATE_OP(endpoint, kv_destroy_session);
    return endpoint->ops->kv_destroy_session(endpoint->ctx, session_id);
}

att1_status_t att1_aimu_conformance_kv_append(att1_aimu_conformance_endpoint *endpoint,
                                              uint64_t session_id,
                                              size_t layer_id,
                                              size_t position,
                                              const float *key,
                                              size_t key_count,
                                              const float *value,
                                              size_t value_count)
{
    ATT1_CONF_VALIDATE_OP(endpoint, kv_append);
    return endpoint->ops->kv_append(endpoint->ctx,
                                    session_id,
                                    layer_id,
                                    position,
                                    key,
                                    key_count,
                                    value,
                                    value_count);
}

att1_status_t att1_aimu_conformance_kv_read(att1_aimu_conformance_endpoint *endpoint,
                                            uint64_t session_id,
                                            size_t layer_id,
                                            size_t head_id,
                                            size_t position,
                                            float *out_key,
                                            size_t key_count,
                                            float *out_value,
                                            size_t value_count)
{
    ATT1_CONF_VALIDATE_OP(endpoint, kv_read);
    return endpoint->ops->kv_read(endpoint->ctx,
                                  session_id,
                                  layer_id,
                                  head_id,
                                  position,
                                  out_key,
                                  key_count,
                                  out_value,
                                  value_count);
}

att1_status_t att1_aimu_conformance_kv_copy_range(att1_aimu_conformance_endpoint *endpoint,
                                                  uint64_t session_id,
                                                  size_t layer_id,
                                                  size_t head_id,
                                                  size_t start_position,
                                                  size_t position_count,
                                                  float *out_keys,
                                                  size_t keys_count,
                                                  float *out_values,
                                                  size_t values_count)
{
    ATT1_CONF_VALIDATE_OP(endpoint, kv_copy_range);
    return endpoint->ops->kv_copy_range(endpoint->ctx,
                                        session_id,
                                        layer_id,
                                        head_id,
                                        start_position,
                                        position_count,
                                        out_keys,
                                        keys_count,
                                        out_values,
                                        values_count);
}

att1_status_t att1_aimu_conformance_kv_get_counters(att1_aimu_conformance_endpoint *endpoint,
                                                    att1_kv_mmu_counters *out)
{
    ATT1_CONF_VALIDATE_OP(endpoint, kv_get_counters);
    return endpoint->ops->kv_get_counters(endpoint->ctx, out);
}

#undef ATT1_CONF_VALIDATE_OP

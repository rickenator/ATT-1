#include "att1.h"

#include <stdio.h>

att1_sim_config att1_default_config(void)
{
    att1_sim_config config = {
        .tile = {
            .tile_id = 0,
            .compute_lanes = 16,
            .local_model_memory_bytes = 256u * 1024u * 1024u,
            .local_scratch_bytes = 16u * 1024u * 1024u,
        },
        .fabric = {
            .tile_count = 1,
            .max_packet_bytes = 256,
            .link_latency_cycles = 32,
            .link_bandwidth_bytes_per_sec = 64ull * 1024ull * 1024ull * 1024ull,
        },
        .kv = {
            .page_bytes = 16u * 1024u,
            .max_pages = 1024,
            .address_bits = 48,
        },
        .random_seed = 1,
    };

    return config;
}

int main(void)
{
    att1_sim_config config = att1_default_config();

    puts("ATT-1 / Aniviza Tensor Tile");
    puts("Phase-1 simulator: Milestone 0");
    printf("tiles=%u local_model_memory=%zu bytes kv_page=%zu bytes\n",
           config.fabric.tile_count,
           config.tile.local_model_memory_bytes,
           config.kv.page_bytes);

    att1_log(ATT1_LOG_INFO, "startup complete");
    return 0;
}

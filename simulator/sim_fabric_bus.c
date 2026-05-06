#include "att1_fabric.h"

int att1_sim_fabric_bus_create(att1_fabric *fabric,
                               size_t tile_count,
                               size_t queue_capacity,
                               size_t max_payload_bytes)
{
    const att1_fabric_bus_config config = {
        .tile_count = tile_count,
        .queue_capacity = queue_capacity,
        .max_payload_bytes = max_payload_bytes
    };

    return att1_fabric_create(fabric, &config);
}

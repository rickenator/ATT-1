#include "att1_tile.h"

#include <string.h>

void att1_tile_state_init(att1_tile_state *state, uint32_t tile_id)
{
    if (state == NULL) {
        return;
    }

    memset(state, 0, sizeof(*state));
    state->tile_id = tile_id;
}

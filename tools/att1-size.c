#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct preset_shape {
    const char *name;
    const char *note;
    uint64_t total_params;
    uint64_t active_params_per_token;
    uint64_t n_layers;
    uint64_t n_heads;
    uint64_t head_dim;
    uint64_t d_model;
    uint64_t max_seq_len;
    uint64_t n_tiles;
} preset_shape;

static uint64_t div_round_up(uint64_t value, uint64_t divisor)
{
    return (value + divisor - 1u) / divisor;
}

static void print_bytes_line(const char *label, uint64_t bytes)
{
    printf("%s=%llu bytes (%.3f GiB)\n",
           label,
           (unsigned long long)bytes,
           (double)bytes / (1024.0 * 1024.0 * 1024.0));
}

static uint64_t tiny_params(void)
{
    const uint64_t vocab = 256u;
    const uint64_t layers = 2u;
    const uint64_t d_model = 4u;
    const uint64_t d_ff = 8u;
    const uint64_t token_embedding = vocab * d_model;
    const uint64_t per_layer =
        d_model +
        (4u * d_model * d_model) +
        d_model +
        (2u * d_model * d_ff) +
        (d_ff * d_model);
    const uint64_t output = d_model + (d_model * vocab);

    return token_embedding + (layers * per_layer) + output;
}

static int load_preset(const char *name, preset_shape *out)
{
    if ((name == NULL) || (out == NULL)) {
        return -1;
    }

    if (strcmp(name, "tiny-dummy") == 0) {
        out->name = "tiny-dummy";
        out->note = "Executable dummy ATT-1 model shape.";
        out->total_params = tiny_params();
        out->active_params_per_token = tiny_params();
        out->n_layers = 2u;
        out->n_heads = 2u;
        out->head_dim = 2u;
        out->d_model = 4u;
        out->max_seq_len = 8u;
        out->n_tiles = 1u;
        return 0;
    }

    if (strcmp(name, "gpt-oss-120b-shape") == 0) {
        out->name = "gpt-oss-120b-shape";
        out->note = "synthetic/non-executable architecture estimate only; no real gpt-oss-120b inference is attempted.";
        out->total_params = 120000000000ull;
        out->active_params_per_token = 5100000000ull;
        out->n_layers = 36u;
        out->n_heads = 64u;
        out->head_dim = 128u;
        out->d_model = out->n_heads * out->head_dim;
        out->max_seq_len = 131072u;
        out->n_tiles = 16u;
        return 0;
    }

    return -1;
}

static void print_report(const preset_shape *shape)
{
    const uint64_t f32_bytes = shape->total_params * 4u;
    const uint64_t f16_bytes = shape->total_params * 2u;
    const uint64_t q8_bytes = shape->total_params;
    const uint64_t q4_bytes = div_round_up(shape->total_params, 2u);
    const uint64_t kv_bytes_f16 =
        shape->n_layers *
        shape->max_seq_len *
        shape->n_heads *
        shape->head_dim *
        2u *
        2u;
    const uint64_t activation_bytes = shape->d_model * 4u;

    printf("preset=%s\n", shape->name);
    printf("note=%s\n", shape->note);
    printf("total_params=%llu\n",
           (unsigned long long)shape->total_params);
    printf("active_params_per_token=%llu\n",
           (unsigned long long)shape->active_params_per_token);
    printf("layers=%llu heads=%llu head_dim=%llu d_model=%llu max_seq_len=%llu tiles=%llu\n",
           (unsigned long long)shape->n_layers,
           (unsigned long long)shape->n_heads,
           (unsigned long long)shape->head_dim,
           (unsigned long long)shape->d_model,
           (unsigned long long)shape->max_seq_len,
           (unsigned long long)shape->n_tiles);
    print_bytes_line("model_bytes_f32", f32_bytes);
    print_bytes_line("model_bytes_f16", f16_bytes);
    print_bytes_line("model_bytes_q8", q8_bytes);
    print_bytes_line("model_bytes_q4", q4_bytes);
    print_bytes_line("estimated_bytes_per_tile_f16",
                     div_round_up(f16_bytes, shape->n_tiles));
    print_bytes_line("kv_bytes_per_session_f16", kv_bytes_f16);
    print_bytes_line("activation_bytes_per_token_f32", activation_bytes);
}

int main(int argc, char **argv)
{
    preset_shape shape;
    const char *preset = NULL;

    if ((argc == 3) && (strcmp(argv[1], "--preset") == 0)) {
        preset = argv[2];
    } else {
        fprintf(stderr,
                "usage: %s --preset tiny-dummy|gpt-oss-120b-shape\n",
                argv[0]);
        return 1;
    }

    if (load_preset(preset, &shape) != 0) {
        fprintf(stderr, "unknown preset: %s\n", preset);
        return 1;
    }

    print_report(&shape);
    return 0;
}

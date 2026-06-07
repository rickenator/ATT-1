# ATT-1 Guided Fuzz Seeds

This directory is the default seed corpus location for optional M152
coverage-guided model-loader fuzzing targets.

The deterministic smoke corpus is still embedded in `tests/fuzz_model_loader.c`
and `compiler/fuzz_json_schemas.py`. Add persistent binary `.att1` seed files
here only when they are small, public, and safe to commit.

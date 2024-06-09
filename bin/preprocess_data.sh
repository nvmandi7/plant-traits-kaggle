#!/usr/bin/env bash
#
# Run unit tests

source bin/setup_environment.sh

docker compose run --rm --entrypoint python app -m src.data.preprocessing.preprocess_tabular_data "$@"
# docker compose run --rm --entrypoint python app -m src.data.preprocessing.precompute_embeddings "$@"

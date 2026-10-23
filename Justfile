set shell := ["bash", "-euc"]

image := "special-topic-etl"
run := 'podman run --rm --userns=keep-id -v "$PWD:/work:Z" -w /work ' + image

# Submission entry point: Justfile -> Containerfile -> Makefile -> src/*.py -> DB -> safetensors.
default: pipeline

# Download, build data/particles.db, then export data/inputs/{train,eval}.safetensors.
pipeline: build
    mkdir -p data
    {{run}} make all

build:
    podman build --file Containerfile --tag {{image}} .

# Verify DB, reload tensors with torch (TensorDataset/DataLoader), run unit tests.
check: build
    {{run}} make check

# Verify cached DB outputs only.
verify: build
    {{run}} make verify

# Rebuild DB and tensors twice; compare DB fingerprint and tensor file hashes.
idempotency: build
    {{run}} make idempotency

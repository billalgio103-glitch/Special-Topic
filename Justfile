set shell := ["bash", "-euc"]

image := "special-topic-etl"
run := 'podman run --rm --userns=keep-id -v "$PWD:/work:Z" -w /work ' + image

# Submission entry point: Justfile -> Containerfile -> Makefile -> src/*.py -> DB -> tensors
# -> data/model/results.json -> report/report.typ -> report/report.pdf.
default: pipeline

# Download, build the DB, export tensors, train (only if inputs changed), write the report PDF.
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

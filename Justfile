set shell := ["bash", "-euc"]

# Default submission entry point: Justfile -> Containerfile -> Makefile -> scripts -> DB.
default:
    podman build --file Containerfile --tag special-topic-etl .
    mkdir -p data
    podman run --rm --userns=keep-id -v "$PWD:/work:Z" -w /work special-topic-etl make all

# Verify cached outputs in the same container.
verify:
    podman run --rm --userns=keep-id -v "$PWD:/work:Z" -w /work special-topic-etl make verify

# Rebuild and compare a deterministic database fingerprint.
idempotency:
    podman run --rm --userns=keep-id -v "$PWD:/work:Z" -w /work special-topic-etl make idempotency

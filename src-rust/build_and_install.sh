#!/bin/bash 

# Build the claurst binary
cargo build --release --package claurst && \
    sudo cp target/release/claurst /usr/local/bin/claurst
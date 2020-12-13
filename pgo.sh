#!/usr/bin/env bash

# Script to compile wfc-rust with profile guided optimization enabled
# See: https://doc.rust-lang.org/rustc/profile-guided-optimization.html for more details

# STEP 0: Make sure there is no left-over profiling data from previous runs
rm -rf /tmp/pgo-data

# STEP 1: Build the instrumented binaries
RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" \
  cargo build \
    --release \
    --target=x86_64-apple-darwin

# STEP 2: Run the instrumented binaries with some typical data or use-case
echo "Running instrumented build"
./target/x86_64-apple-darwin/release/wfc-rust

# STEP 3: Merge the `.profraw` files into a `.profdata` file
llvm-profdata merge -o /tmp/pgo-data/merged.profdata /tmp/pgo-data

# STEP 4: Use the `.profdata` file for guiding optimizations
RUSTFLAGS="-Cprofile-use=/tmp/pgo-data/merged.profdata" \
  cargo build \
    --release \
    --target=x86_64-apple-darwin

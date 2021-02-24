#!/usr/bin/env bash

rm -rf target/instruments/wfc_rust.trace/
cargo instruments --release --out target/instruments/wfc_rust.trace \
  && open target/instruments/wfc_rust.trace

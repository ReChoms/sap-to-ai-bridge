#!/bin/bash

# Exit immediately if compilation fails
set -e

echo "======================================"
echo "Compiling Release Binary..."
echo "======================================"
cargo build --release

echo ""
echo "======================================"
echo "Starting Benchmark Experiment"
echo "======================================"

for BATCH in 64 128 256 512 1024
do
    echo ""
    echo ">>> Testing Batch Size: $BATCH"
    echo "--------------------------------------"
    
    # Execute the raw compiled binary, bypassing cargo overhead
    time target/release/sap-to-ai-bridge ingest -f data/kna1.csv -o -b $BATCH
done

echo ""
echo "======================================"
echo "Benchmark Complete"
echo "======================================"

# APUF Model Decoder: Approach

## Problem Overview
This project decodes the parameters of an XOR-APUF model from encoded weight vectors. The goal is to recover underlying APUF parameters and reconstruct the original XOR-APUF model while minimizing reconstruction error.

## Approach
1. Load the encoded APUF weight vectors from `data/public_mod.txt` or `data/secret/secret_mod.txt`.
2. For each weight vector, extract the two factorized APUF components using the Kronecker product structure.
3. Recover parameter relationships from the extracted factor vectors.
4. Reconstruct the original XOR-APUF model using decoded parameters.
5. Evaluate reconstruction quality using the L2 norm between original and reconstructed weight vectors.

## Solution Design
- `APUFDecoder.get_APUF_model`: builds an APUF weight vector from delay parameter values.
- `APUFDecoder.get_XOR_APUF_model`: generates XOR-APUF weights with the Kronecker product of two APUF weight vectors.
- `APUFDecoder.my_decode`: decodes parameters from an XOR-APUF weight vector.
- `APUFDecoder.reconstruct_and_evaluate`: reconstructs and measures error.

## Result Files
- `results/decoding_results.txt`: summary of average timing and reconstruction error.
- `results/reconstruction_errors.txt`: per-model reconstruction errors.
- `results/timing_analysis.txt`: per-model decoding times.

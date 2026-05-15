# Data Directory

This folder contains the input weight vectors used for APUF model decoding.

## Files
- `public_mod.txt`: Public XOR-APUF weight vectors available for development and testing.
- `secret/secret_mod.txt`: Private secret weight vectors used for final evaluation when available.

## Format
- Each row corresponds to one XOR-APUF weight vector.
- Each row contains numeric values separated by whitespace.
- The number of columns represents the combined dimension of the XOR-APUF model.

## Usage
The main script `solution.py` loads the public file by default and will use the secret file if present.

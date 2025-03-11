# LDPC Code Simulator

A Python implementation of Low-Density Parity-Check (LDPC) codes for error correction in noisy communication channels. This simulator provides tools for generating sparse parity-check matrices, encoding messages, simulating AWGN channel noise, and decoding using the sum-product algorithm with improved numerical stability. Features include BER (Bit Error Rate) performance analysis across different SNR levels and visualization capabilities.

## Dependencies
- Python 3.x
- NumPy
- Matplotlib

## Usage
Simply run the main script to execute a simulation with default parameters:
```bash
python ldpc_sim.py
```

This will:
1. Create an LDPC code with n=20, k=10 (codeword length 20, message length 10)
2. Generate parity-check and generator matrices
3. Encode a random test message
4. Simulate transmission over an AWGN channel at various SNR levels
5. Decode using the sum-product algorithm
6. Display a BER vs SNR performance plot

You can modify simulation parameters by changing the values in the `run_simulation()` function.

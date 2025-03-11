import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

class LDPCSimulator:
    def __init__(self, n: int = 20, k: int = 10):
        """
        Initialize LDPC Simulator with code parameters

        Args:
            n (int): Codeword length
            k (int): Message length
        """
        self.n = n  # codeword length
        self.k = k  # message length
        self.m = n - k  # number of parity checks
        self.H = None  # parity check matrix
        self.G = None  # generator matrix

    def generate_parity_matrix(self) -> np.ndarray:
        """
        Generate a random sparse parity-check matrix with better properties
        """
        # Create sparse matrix with weight 3 per row and roughly weight 3 per column
        H = np.zeros((self.m, self.n))
        col_weights = np.zeros(self.n)

        # Ensure each row has exactly 3 ones
        for i in range(self.m):
            # Try to distribute ones evenly across columns
            available_cols = np.where(col_weights < 4)[0]
            if len(available_cols) < 3:
                available_cols = np.arange(self.n)

            # Randomly place 3 ones in each row
            ones_position = np.random.choice(available_cols, 3, replace=False)
            H[i, ones_position] = 1
            col_weights[ones_position] += 1

        self.H = H
        return H

    def generate_generator_matrix(self) -> np.ndarray:
        """
        Construct generator matrix from parity-check matrix using improved systematic form conversion
        """
        H_temp = self.H.copy()
        n = self.n
        k = self.k
        m = self.m

        # Try multiple times to get systematic form
        max_attempts = 100
        for attempt in range(max_attempts):
            H_working = H_temp.copy()
            success = True

            # Rearrange H to systematic form [P | I]
            for i in range(m):
                # Find pivot
                pivot_found = False
                for j in range(i, n):
                    if H_working[i, j] == 1:
                        if j != i + k:
                            # Swap columns
                            H_working[:, [j, i + k]] = H_working[:, [i + k, j]]
                        pivot_found = True
                        break

                if not pivot_found:
                    success = False
                    break

                # Eliminate ones in the column
                for l in range(m):
                    if l != i and H_working[l, i + k] == 1:
                        H_working[l, :] = (H_working[l, :] + H_working[i, :]) % 2

            if success:
                # Extract P matrix
                P = H_working[:, :k]

                # Create generator matrix G = [I | P^T]
                G = np.zeros((k, n))
                G[:, :k] = np.eye(k)
                G[:, k:] = P.T

                self.G = G
                return G

            # If failed, generate new parity check matrix and try again
            if attempt < max_attempts - 1:
                H_temp = self.generate_parity_matrix()

        raise ValueError("Could not create systematic form after maximum attempts")

    def encode_message(self, message: np.ndarray) -> np.ndarray:
        """
        Encode a message using the generator matrix

        Args:
            message (np.ndarray): Binary message vector

        Returns:
            np.ndarray: Encoded codeword
        """
        return np.mod(np.dot(message, self.G), 2)

    def add_noise(self, codeword: np.ndarray, snr_db: float) -> np.ndarray:
        """
        Add AWGN noise to the codeword

        Args:
            codeword (np.ndarray): Binary codeword
            snr_db (float): Signal-to-noise ratio in dB

        Returns:
            np.ndarray: Noisy received signal
        """
        # Convert to BPSK (+1/-1)
        bpsk = 1 - 2 * codeword

        # Calculate noise power
        snr = 10 ** (snr_db / 10)
        noise_power = 1 / (2 * snr)

        # Add Gaussian noise
        noise = np.sqrt(noise_power) * np.random.randn(len(codeword))
        received = bpsk + noise

        return received

    def decode_received(self, received: np.ndarray, max_iterations: int = 50) -> Tuple[np.ndarray, int]:
        """
        Decode received signal using sum-product algorithm with improved numerical stability

        Args:
            received (np.ndarray): Received noisy signal
            max_iterations (int): Maximum number of iterations

        Returns:
            Tuple[np.ndarray, int]: (Decoded message, number of iterations)
        """
        # Initialize LLRs (Log-Likelihood Ratios)
        llr = 2 * received

        # Initialize messages
        check_to_bit = np.zeros((self.m, self.n))
        bit_to_check = np.zeros((self.m, self.n))
        prev_beliefs = None

        # Get positions of ones in H
        H_indices = [[] for _ in range(self.m)]
        for i in range(self.m):
            H_indices[i] = np.where(self.H[i] == 1)[0]

        # Iterate
        for iteration in range(max_iterations):
            # Update bit-to-check messages
            for i in range(self.m):
                for j in H_indices[i]:
                    bit_to_check[i, j] = llr[j] - check_to_bit[i, j]

            # Normalize bit messages
            bit_scale = np.maximum(1.0, np.max(np.abs(bit_to_check)))
            bit_to_check /= bit_scale

            # Update check-to-bit messages with improved numerical stability
            for i in range(self.m):
                for j in H_indices[i]:
                    other_bits = [b for b in H_indices[i] if b != j]
                    signs = np.sign(bit_to_check[i, other_bits])
                    magnitudes = np.minimum(np.abs(bit_to_check[i, other_bits]), 15)

                    # Calculate message using more stable operations
                    sign = np.prod(signs)
                    magnitude = np.sum(np.log(np.tanh(magnitudes / 2)))
                    check_to_bit[i, j] = sign * 2 * np.arctanh(np.minimum(np.exp(magnitude), 0.99))

                    # Handle numerical instabilities
                    if np.isnan(check_to_bit[i, j]) or np.isinf(check_to_bit[i, j]):
                        check_to_bit[i, j] = sign * 15.0

            # Normalize check messages
            check_scale = np.maximum(1.0, np.max(np.abs(check_to_bit)))
            check_to_bit /= check_scale

            # Compute beliefs
            beliefs = llr.copy()
            for i in range(self.m):
                for j in H_indices[i]:
                    beliefs[j] += check_to_bit[i, j]

            # Make hard decision
            decoded = (beliefs < 0).astype(int)

            # Check if valid codeword
            syndrome = np.mod(np.dot(self.H, decoded), 2)
            if np.all(syndrome == 0):
                return decoded, iteration + 1

            # Check for convergence
            if prev_beliefs is not None and iteration % 5 == 0:
                belief_change = np.max(np.abs(beliefs - prev_beliefs))
                if belief_change < 0.01:
                    return decoded, iteration + 1

            prev_beliefs = beliefs.copy()

        return decoded, max_iterations

def run_simulation():
    """
    Run LDPC simulation and plot results
    """
    # Initialize simulator
    sim = LDPCSimulator(n=20, k=10)

    # Generate matrices
    sim.generate_parity_matrix()
    sim.generate_generator_matrix()

    # Test message
    message = np.random.randint(0, 2, sim.k)

    # SNR range for testing
    snr_range = np.arange(0, 11, 2)
    ber_results = []

    print("Running simulation...")
    print(f"Message: {message}")

    for snr in snr_range:
        errors = 0
        n_trials = 50  # Reduced number of trials

        for _ in range(n_trials):
            # Encode
            codeword = sim.encode_message(message)

            # Add noise
            received = sim.add_noise(codeword, snr)

            # Decode
            decoded, _ = sim.decode_received(received)

            # Count errors in message bits
            errors += np.sum(decoded[:sim.k] != message)

        ber = errors / (n_trials * sim.k)
        ber_results.append(ber)
        print(f"SNR: {snr} dB, BER: {ber:.4f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_range, ber_results, 'bo-')
    plt.grid(True)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate')
    plt.title('LDPC Code Performance')
    plt.show()

if __name__ == "__main__":
    run_simulation()







                     
                      

          

          




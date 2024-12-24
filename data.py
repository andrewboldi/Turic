#!/usr/bin/python3
from cryptography.hazmat.primitives.asymmetric import rsa
import multiprocessing

# Function to generate an RSA private key and extract p and q
def generate_and_print_key(index):
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=512
    )
    p = str(private_key.private_numbers().p)
    q = str(private_key.private_numbers().q)

    with open(f"data/{str(index).zfill(6)}.txt", "a") as _file:
        _file.write(p + "\n")
        _file.write(q + "\n")
        _file.write(str(int(p) * int(q)))
#    print(f"Key {index} - p: {p}, q: {q}")

if __name__ == "__main__":
    num_cores = 24  # Number of CPU cores to utilize

    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=num_cores)

    # Generate keys concurrently using multiple cores
    keys_indices = range(1000000)  # Generate 1000000 keys
    pool.map(generate_and_print_key, keys_indices)

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()

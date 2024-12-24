import os
import tqdm
arr = os.listdir("data/test/decoded")

for i in tqdm.tqdm(range(0, 50000)):
    if f"p{i}.txt" in arr:
        continue
    os.system(f"openssl genpkey -algorithm RSA -out data/test/private/p{i}.pem -pkeyopt rsa_keygen_bits:2048 -quiet && openssl rsa -text -in data/test/private/p{i}.pem >> data/test/decoded/p{i}.txt")

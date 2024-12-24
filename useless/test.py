from OpenSSL import crypto

pk = crypto.PKey()
pk.generate_key(crypto.TYPE_RSA, 2048)

print(f"Private key : {crypto.dump_privatekey(crypto.FILETYPE_PEM, pk)}\n")
print(f"Public key : {crypto.dump_publickey(crypto.FILETYPE_PEM, pk)}\n")

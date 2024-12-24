#!/bin/sh
max = 5
for i in `seq 6 $max`
do
	openssl genpkey -algorithm RSA -out p${i}.pem -pkeyopt rsa_keygen_bits:2048 && openssl rsa -pubout -in p${i}.pem -out ${i}p.pem
done

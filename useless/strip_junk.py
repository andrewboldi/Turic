import os

for _file in os.listdir("decoded/"):
    print(f"opening {_file}")
    arr = open("decoded/" + _file).readlines()
    arr = arr[22:60]
    for i, x in enumerate(arr):
        arr[i] = arr[i][:-1]
        arr[i] = arr[i].replace("   ","")
        arr[i] = arr[i][1:]

    arr.remove("rime1:")
    arr.remove("rime2:")
    for i, x in enumerate(arr):
        arr[i] = arr[i].replace(":","")

    prod = int(bin(int("".join(arr[:18]), 16)).replace("0b", ""))
    p1 = int(bin(int("".join(arr[18:27]), 16)).replace("0b", ""))
    p2 = int(bin(int("".join(arr[27:]), 16)).replace("0b", ""))

    print("============================================================")
    print(prod)
    print("============================================================")
    print(p1)
    print("============================================================")
    print(p2)

    exit(0)
    

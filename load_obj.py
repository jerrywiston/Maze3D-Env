import numpy as np

def load(path='./resource/obj/Cube.obj'):
    file = open(path)
    while True:
        content = file.readline()
        if content == "":
            return
        temp = content.strip().split(" ")
        if temp[0] == "v":
            print(temp)
        elif temp[0] == "vn":
            print(temp)
        elif temp[0] == "vt":
            print(temp)
        elif temp[0] == "f":
            print(temp)
        
load()
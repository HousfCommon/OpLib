#--just practice--
import numpy as np

def main():
    n = 20
    a = 0
    b = 1
    c = 0
    for i in range(n):
        if i==0:
            c = 0
        elif i==1:
            c = b
        else:
            c = b + a
            a = b
            b = c

    print(c)


if __name__ == '__main__':
    main()







if __name__=='__main__':
    main()

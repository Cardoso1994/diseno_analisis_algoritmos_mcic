#!/usr/bin/env python3
###############################################################################
#
# author: Marco Antonio Cardoso Moreno
#
# Funciones FFT e IFFT
#
###############################################################################

import numpy as np

def fft(x):
    num = len(x)

    if num <= 1:
        return x

    even = fft(x[0::2])
    odd = fft(x[1::2])

    y = np.zeros((num,), dtype=np.complex128)
    for k in range(num // 2):
        t = np.exp(-2j * np.pi * k / num) * odd[k]
        y[k] = even[k] + t
        y[k + num // 2] = even[k] - t

    return y

def ifft(x):
    num = len(x)

    if num <= 1:
        return x

    even = ifft(x[0::2])
    odd = ifft(x[1::2])
    y = np.zeros((num,), dtype=np.complex128)
    for k in range(num // 2):
        t = np.exp(2j * np.pi * k / num) * odd[k]
        y[k] = (even[k] + t) / num
        y[k + num // 2] = (even[k] - t) / num

    return y

if __name__ == '__main__':
    x = np.exp(2j * np.pi * np.arange(8) / 8)
    print("Numpy FFT:")
    print(np.fft.fft(x))
    print("Own FFT:")
    print(fft(x))
    print()
    print("Numpy IFFT:")
    print(np.fft.ifft(x))
    print("Own IFFT:")
    print(ifft(x))

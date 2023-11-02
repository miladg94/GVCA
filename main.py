import math
import numpy as np
from numba import jit, cuda
from timeit import default_timer as timer
import DCTTransformNative as dct
import torch
import torch_dct as tdct

# C:\Users\rnili\OneDrive\Desktop\VCA-stable\test.yuv
#stream = input("Enter path to input file: ")
#w = input("Enter width of video: ")
#h = input("Enter height of video: ")
#frame_count = input("Enter frame count of video: ")

stream = r'C:\Users\rnili\OneDrive\Desktop\VCA-stable\test.yuv'
w = 1920
h = 1080
block_size = 32
height = (math.ceil(h / block_size)) * block_size
width = (math.ceil(w / block_size)) * block_size
Y = np.zeros((height, width), dtype=np.uint8)
hpad = height - h
wpad = width - w
frame_count = 600
frame_size = int(1.5 * w * h)


@jit(target_backend='cuda', nopython=True)
def performDCT(Y, block_size, width, height):
    blocks = []
    bCount = 0
    for a in range(0, height, block_size):
        for b in range(0, width, block_size):
            block = Y[a:a + block_size, b:b + block_size]
            block = block.flatten()
            block = block.astype('int32')
            blocks.append(block)
            bCount += 1
    frameTex = 0
    bEnergy = []
    for i in range(bCount):
        buffer = dct.partial_butterfly32(blocks[i], 4, 32)
        DCT = dct.partial_butterfly32(buffer, 11, 32)
        WDCT = np.abs(DCT) * dct.weights_dct32 // 256
        sumW = sum(WDCT)
        bEnergy.append(sumW)
        frameTex += sumW
    avgEnergy = frameTex // (bCount * 90)
    return avgEnergy, bEnergy, bCount


#@jit(target_backend='cuda', nopython=True)
#def gpu(Y, block_size, width, height):
#    avgEnergy, bEnergy, bCount = performDCT(Y, block_size, width, height)
#    return avgEnergy, bEnergy, bCount


def main():
    avg1 = np.zeros(frame_count)
    avg2 = np.zeros(frame_count)
    avgsE = []
    avgsH = []
    with open(stream, 'rb') as file:
        start = timer()
        for f in range(frame_count):
            frame_data = file.read(frame_size)
            _Y = np.frombuffer(frame_data[:w * h], dtype=np.uint8).reshape(h, w)
            Y[:h, :w] = _Y
            for hp in range(hpad):
                Y[h + hp, :] = Y[h - 1, :]
            for wp in range(wpad):
                Y[:, w + wp] = Y[:, w - 1]
            fTime = timer()
            avgE, bEnergy, bCount = performDCT(Y, block_size, width, height)
            avgsE.append(avgE)
            avg1 = np.resize(avg1, bCount)
            avg2 = bEnergy.copy()
            TC = np.abs(avg2 - avg1)
            avg1 = avg2.copy()
            avgTC = sum(TC) / (len(TC) * 18)
            avgsH.append(avgTC)
            avgsH[0] = 0
            print(f, avgsE[f], avgsH[f], timer() - fTime)
        print(" GPU Total Time:", timer() - start)
    return avgsE, avgsH


if __name__ == "__main__":
    main()





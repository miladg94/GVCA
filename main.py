import numpy as np
from numba import jit, cuda
from timeit import default_timer as timer
import DCTTransformNative as dct
import math

@jit(target_backend='cuda', nopython=True)
def performDCT(Y, block_size, width, height):
    block = np.zeros((block_size, block_size), dtype=np.uint8)
    blocks = []
    blocks_count = 0
    for a in range(0, height, block_size):
        for b in range(0, width, block_size):
            block = Y[a:a + block_size, b:b + block_size]
            # FIX PADDING !
            #for c in range (0, a + block_size - height):
            #    block[height-a+c,:] = block[height-a-1,:]
            #for d in range (0, b + block_size - width):
            #    block[:, width-b+d] = block[:, width-b-1]
            block = block.flatten()
            block = block.astype('int32')
            blocks.append(block)
            blocks_count += 1
    avgs = []

    for i in range(blocks_count):
        buffer = dct.partial_butterfly32(blocks[i], 4, 32)
        DCT = dct.partial_butterfly32(buffer, 11, 32)
        avg = sum(DCT) / len(DCT)
        avgs.append(avg)
    return avgs


def cpu(Y, block_size, width, height):
    averages = performDCT(Y, block_size, width, height)
    return averages
@jit(target_backend='cuda', nopython=True)
def gpu(Y, block_size, width, height):
    averages = performDCT(Y, block_size, width, height)
    return averages


def main():
    stream = input("Enter path to input file: ")
    # C:\Users\rnili\OneDrive\Desktop\VCA-stable\test.yuv
    w = 1920
    h = 1080
    block_size = 32
    frame_count = 120*5
    frame_size = int(1.5 * w * h)
    totalAverageEnergy = []
    with open(stream, 'rb') as file:
        go = timer()
        for f in range(frame_count):
            frame_data = file.read(frame_size)
            height = (math.ceil(h / block_size)) * block_size
            width = (math.ceil(w / block_size)) * block_size
            hpad = height - h
            wpad = width - w
            Y = np.frombuffer(frame_data[:width * height], dtype=np.uint8).reshape(height, width)
            Y = Y.copy()
            for hp in range(hpad):
                Y[h + hp] = Y[h - 1]
            for wp in range(wpad):
                Y[w + wp] = Y[w - 1]
            start = timer()
            averageEnergy = gpu(Y, block_size, width, height)
            totalAverageEnergy.append(averageEnergy)
            print("GPU Frame Time:", f, timer() - start)
        print(" GPU Total Time:", timer() - go)
    return totalAverageEnergy


if __name__ == "__main__":
    main()





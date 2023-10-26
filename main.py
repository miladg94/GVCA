import numpy as np
from numba import jit, cuda
from timeit import default_timer as timer
import DCTTransformNative as dct
import math

@jit(target_backend='cuda', nopython=True)
def performDCT(Y, block_size, width, height):
    blocks = []
    blocks_count = 0
    for a in range(0, height, block_size):
        for b in range(0, width, block_size):
            block = Y[a:a + block_size, b:b + block_size]
            block = block.flatten()
            block = block.astype('int32')
            blocks.append(block)
            blocks_count += 1
    avgs = []
    for i in range(blocks_count):
        buffer = dct.partial_butterfly32(blocks[i], 4, 32)
        DCT = dct.partial_butterfly32(buffer, 11, 32)
        WDCT = np.abs(DCT) * dct.weights_dct32 / 256
        avg = sum(WDCT) / len(WDCT)
        avgs.append(avg)
    return sum(avgs)/len(avgs)


@jit(target_backend='cuda', nopython=True)
def gpu(Y, block_size, width, height):
    averages = performDCT(Y, block_size, width, height)
    return averages


def main():
    stream = r'C:\Users\rnili\OneDrive\Desktop\VCA-stable\test.yuv'
    #stream = input("Enter path to input file: ")
    # C:\Users\rnili\OneDrive\Desktop\VCA-stable\test.yuv
    w = 1920
    h = 1080
    block_size = 32
    frame_count = 600
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
                Y[h + hp, :] = Y[h - 1, :]
            for wp in range(wpad):
                Y[:, w + wp] = Y[:, w - 1]
            start = timer()
            averageEnergy = gpu(Y, block_size, width, height)
            totalAverageEnergy.append(averageEnergy)
            print("GPU Frame Time:", f, timer() - start)
        print(" GPU Total Time:", timer() - go)
    return totalAverageEnergy


if __name__ == "__main__":
    main()





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
    frameTex = 0
    for i in range(blocks_count):
        buffer = dct.partial_butterfly32(blocks[i], 4, 32)
        DCT = dct.partial_butterfly32(buffer, 11, 32)
        WDCT = np.abs(DCT) * dct.weights_dct32 // 256
        sumW = sum(WDCT)
        avg = sumW/len(WDCT)
        avgs.append(avg)
        frameTex += sumW
    return frameTex // (blocks_count * 90), avgs, blocks_count


@jit(target_backend='cuda', nopython=True)
def gpu(Y, block_size, width, height):
    avgEnergy, blocksAVG, blkcnt = performDCT(Y, block_size, width, height)
    return avgEnergy, blocksAVG, blkcnt


def main():
    stream = r'C:\Users\rnili\OneDrive\Desktop\VCA-stable\test.yuv'
    #stream = input("Enter path to input file: ")
    # C:\Users\rnili\OneDrive\Desktop\VCA-stable\test.yuv

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

    avgsE = []
    avg1 = np.zeros(frame_count)
    avg2 = np.zeros(frame_count)
    tempComps = []

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

            Ftime = timer()
            avgE, blocksAVG, blocks_count = gpu(Y, block_size, width, height)
            avgsE.append(avgE)

            avg1 = np.resize(avg1, blocks_count)
            avg2 = blocksAVG.copy()
            tempComp = np.abs(avg2 - avg1)
            tempComps.append(tempComp)
            avg1 = avg2.copy()

            #print(f, avgsE[f])
            print("GPU Frame Time:", f, timer() - Ftime)
        print(" GPU Total Time:", timer() - start)
    return totalAverageEnergy, tempComps


if __name__ == "__main__":
    main()





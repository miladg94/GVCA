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
    return frameTex // (blocks_count * 90), avgs


@jit(target_backend='cuda', nopython=True)
def gpu(Y, block_size, width, height):
    avgEnergy, blocksAVG = performDCT(Y, block_size, width, height)
    return avgEnergy, blocksAVG


def main():
    stream = r'C:\Users\rnili\OneDrive\Desktop\VCA-stable\0001_426x240.yuv'
    #stream = input("Enter path to input file: ")
    # C:\Users\rnili\OneDrive\Desktop\VCA-stable\test.yuv
    w = 426
    h = 240
    block_size = 32
    frame_count = 2
    frame_size = int(1.5 * w * h)
    totalAverageEnergy = []
    #bAVG1 = np.zeros(frame_count)
    #bAVG2 = np.zeros(frame_count)
    #tempComps = np.zeros(frame_count)
    with open(stream, 'rb') as file:
        go = timer()
        for f in range(frame_count):
            frame_data = file.read(frame_size)
            _Y = np.frombuffer(frame_data[:w * h], dtype=np.uint8).reshape(h, w)
            height = (math.ceil(h / block_size)) * block_size
            width = (math.ceil(w / block_size)) * block_size
            hpad = height - h
            wpad = width - w
            Y = np.zeros((height, width), dtype=_Y.dtype)
            Y[:h, :w] = _Y
            for hp in range(hpad):
                Y[h + hp, :] = Y[h - 1, :]
            for wp in range(wpad):
                Y[:, w + wp] = Y[:, w - 1]
            start = timer()
            averageEnergy, blocksAVG = gpu(Y, block_size, width, height)
            totalAverageEnergy.append(averageEnergy)
            #bAVG2 = blocksAVG.copy()
            #tempComp = np.abs(bAVG2 - bAVG1)
            #tempComp [f-1] = tempComp [f]
            #bAVG1 = bAVG2.copy()
            #tempComps.append(tempComp)
            print(f, totalAverageEnergy[f])
            #print("GPU Frame Time:", f, timer() - start)
        print(" GPU Total Time:", timer() - go)
    return totalAverageEnergy


if __name__ == "__main__":
    main()





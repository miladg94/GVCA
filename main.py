import numpy as np
from numba import jit, cuda
from timeit import default_timer as timer
import DCTTransformNative as dct

@jit(target_backend='cuda', nopython=True)
def performDCT(Y, block_size, width, height):
    block = np.zeros((block_size, block_size), dtype=np.uint8)
    blocks = []
    blocks_count = 0
    for a in range(0, height, block_size):
        for b in range(0, width, block_size):
            block = Y[a:a + block_size, b:b + block_size]
            # fill zero values!!!
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
    filename = input("Enter path to input file: ")
    # C:\Users\rnili\OneDrive\Desktop\VCA-stable\test.yuv
    width = 1920
    height = 1080
    block_size = 32
    frame_size = int(1.5 * width * height)
    with open(filename, 'rb') as file:
        frame_data = file.read(frame_size)
    Y = np.frombuffer(frame_data[:width * height], dtype=np.uint8).reshape(height, width)

    #start = timer()
    #averageEnergy = cpu(Y, block_size, width, height)
    #print("CPU Time:", timer() - start)

    start = timer()
    averageEnergy = gpu(Y, block_size, width, height)
    print("GPU Time:", timer() - start)


if __name__ == "__main__":
    main()





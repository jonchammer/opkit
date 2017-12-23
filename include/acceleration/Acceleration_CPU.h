#ifndef ACCELERATION_CPU_H
#define ACCELERATION_CPU_H

#include <cassert>

namespace opkit
{

template <class T>
struct Acceleration_CPU
{
    static void mmMultiply(const T* A, const T* B, T* C,
        const size_t M, const size_t N, const size_t K,
        const size_t ldA, const size_t ldB, const size_t ldC,
        const T alpha, const T beta)
    {
        for(size_t i = 0; i < M; ++i)
        {
            for(size_t j = 0; j < N; ++j)
            {
                T &v = C[i * ldC + j];
                v *= beta;
                for(size_t k = 0; k < K; ++k)
                {
                    v += alpha * A[i * ldA + k] * B[k * ldB + j];
                }
            }
        }
    }

    static void mmtMultiply(const T* A, const T* B, T* C,
        const size_t M, const size_t N, const size_t K,
        const size_t ldA, const size_t ldB, const size_t ldC,
        const T alpha, const T beta)
    {
        for(size_t i = 0; i < M; ++i)
        {
            for(size_t j = 0; j < N; ++j)
            {
                T &v = C[i * ldC + j];
                v *= beta;
                for(size_t k = 0; k < K; ++k)
                {
                    v += alpha * A[i * ldA + k] * B[j * ldB+ k];
                }
            }
        }
    }

    static void mtmMultiply(const T* A, const T* B, T* C,
        const size_t M, const size_t N, const size_t K,
        const size_t ldA, const size_t ldB, const size_t ldC,
        const T alpha, const T beta)
    {
        for(size_t i = 0; i < M; ++i)
        {
            for(size_t j = 0; j < N; ++j)
            {
                T &v = C[i * ldC + j];
                v *= beta;
                for(size_t k = 0; k < K; ++k)
                {
                    v += alpha * A[k * ldA + i] * B[k * ldB + j];
                }
            }
        }
    }

    static void mvMultiply(const T* A, const T* x, T* y,
        const size_t M, const size_t N,
        const T alpha, const T beta,
        const int xInc, const int yInc)
    {
        int yIndex = 0;
        for (size_t i = 0; i < M; ++i)
        {
            T sum{};
            size_t aIndex = i * N;
            int xIndex    = 0;

            for (size_t j = 0; j < N; ++j)
            {
                sum    += A[aIndex + j] * x[xIndex];
                xIndex += xInc;
            }

            y[yIndex] = alpha * sum + beta * y[yIndex];
            yIndex   += yInc;
        }
    }

    static void mtvMultiply(const T* A, const T* x, T* y,
        const size_t M, const size_t N,
        const T alpha, const T beta,
        const int xInc, const int yInc)
    {
        int yIndex = 0;
        for (size_t i = 0; i < N; ++i)
        {
            T sum{};
            int xIndex = 0;

            for (size_t j = 0; j < M; ++j)
            {
                sum    += A[j * N + i] * x[xIndex];
                xIndex += xInc;
            }

            y[yIndex] = alpha * sum + beta * y[yIndex];
            yIndex   += yInc;
        }
    }

    static void symmetricMvMultiply(const T* A, const T* x, T* y,
        const size_t N, const T alpha, const T beta,
        const int xInc, const int yInc)
    {
        int yIndex = 0;
        for (size_t i = 0; i < N; ++i)
        {
            T sum{};
            size_t aIndex = i * N;
            int xIndex    = 0;

            for (size_t j = 0; j < N; ++j)
            {
                sum    += A[aIndex + j] * x[xIndex];
                xIndex += xInc;
            }

            y[yIndex] = alpha * sum + beta * y[yIndex];
            yIndex   += yInc;
        }
    }

    static void outerProduct(const T* x, const T* y, T* A,
        const size_t M, const size_t N, const T alpha,
        const int xInc, const int yInc)
    {
        int aIndex = 0;
        for (size_t i = 0; i < M; ++i)
        {
            int xIndex = 0;
            int yIndex = 0;
            for (size_t j = 0; j < N; ++j)
            {
                A[aIndex] += x[xIndex] * y[yIndex];
                ++aIndex;
                yIndex += yInc;
            }
            xIndex += xInc;
        }
    }

    static void vAdd(const T* x, T* y,
        const size_t N, const T alpha,
        const int xInc, const int yInc)
    {
        int xIndex = 0;
        int yIndex = 0;
        for (size_t i = 0; i < N; ++i)
        {
            y[yIndex] += alpha * x[xIndex];
            xIndex += xInc;
            yIndex += yInc;
        }
    }

    static void vScale(T* x, const T alpha,
        const size_t N, const int xInc)
    {
        int xIndex = 0;
        for (size_t i = 0; i < N; ++i)
        {
            x[xIndex] *= alpha;
            xIndex += xInc;
        }
    }

    static size_t vMaxIndex(const T* x, const size_t N, const int xInc)
    {
        size_t maxIndex = 0;
        T max           = x[0];
        int xIndex      = 0;

        for (size_t i = 0; i < N; ++i)
        {
            if (x[xIndex] > max)
            {
                maxIndex = xIndex;
                max      = x[xIndex];
            }
            xIndex += xInc;
        }

        return maxIndex;
    }

    static void vCopy(const T* x, T* y, const size_t N,
        const int xInc, const int yInc)
    {
        int xIndex = 0;
        int yIndex = 0;

        for (size_t i = 0; i < N; ++i)
        {
            y[yIndex] = x[xIndex];
            xIndex += xInc;
            yIndex += yInc;
        }
    }


    static void matrixAdd(const T* A, T* C,
        const size_t M, const size_t N,
        const size_t aStrideX, const size_t aStrideY,
        const size_t cStrideX, const size_t cStrideY,
        const T alpha, const T beta)
    {
        const T* aPtr = A;
              T* cPtr = C;
        for (size_t i = 0; i < M; ++i)
        {
            axpby(aPtr, cPtr, N, alpha, beta, aStrideX, cStrideX);
            aPtr += aStrideY;
            cPtr += cStrideY;
        }
    }

    static void axpby(const T* x, T* y, const size_t N, const T alpha,
        const T beta, const size_t xInc, const size_t yInc)
    {
        // y = alpha * x + beta * y
        for (size_t i = 0; i < N; ++i)
        {
            *y = alpha * *x + beta * *y;

            x += xInc;
            y += yInc;
        }
    }

    static void im2col(const T* src,
        const size_t srcWidth, const size_t srcHeight, const size_t channels,
        const size_t windowWidth, const size_t windowHeight,
        const size_t xPad, const size_t yPad,
        const size_t xStride, const size_t yStride,
        const size_t xDilation, const size_t yDilation, T* dest)
    {
        const int outputHeight = (srcHeight + 2 * yPad -
            (yDilation * (windowHeight - 1) + 1)) / yStride + 1;
        const int outputWidth = (srcWidth + 2 * xPad -
            (xDilation * (windowWidth - 1) + 1)) / xStride + 1;
        const int channelSize = srcHeight * srcWidth;

        // Handle the data for each channel independently
        for (int channel = channels; channel--; src += channelSize)
        {
            // Iterate over the horizontal and vertical windowed regions
            for (int kernelRow = 0; kernelRow < windowHeight; kernelRow++)
            {
                for (int kernelCol = 0; kernelCol < windowWidth; kernelCol++)
                {
                    // Copy the data inside this window into dest
                    int srcRow = -yPad + kernelRow * yDilation;
                    for (int destRows = outputHeight; destRows; destRows--)
                    {
                        if (!(srcRow >= 0 && srcRow < srcHeight))
                        {
                            for (int destCol = outputWidth; destCol; destCol--)
                                *(dest++) = T{};
                        }
                        else
                        {
                            int srcCol = -xPad + kernelCol * xDilation;
                            for (int destCol = outputWidth; destCol; destCol--)
                            {
                                if (srcCol >= 0 && srcCol < srcWidth)
                                    *(dest++) = src[srcRow * srcWidth + srcCol];
                                else
                                    *(dest++) = T{};

                                srcCol += xStride;
                            }
                        }
                        srcRow += yStride;
                    }
                }
            }
        }
    }

    static void col2im(const T* src,
        const int destWidth, const int destHeight, const int channels,
        const int windowWidth, const int windowHeight,
        const int xPad, const int yPad,
        const int xStride, const int yStride,
        const int xDilation, const int yDilation,
        T* dest)
    {
        std::fill(dest, dest + destHeight * destWidth * channels, T{});

        const int srcHeight = (destHeight + 2 * yPad -
            (yDilation * (windowHeight - 1) + 1)) / yStride + 1;
        const int srcWidth = (destWidth + 2 * xPad -
            (xDilation * (windowWidth - 1) + 1)) / xStride + 1;
        const int channelSize = destHeight * destWidth;

        // Handle the data for each channel independently
        for (int channel = channels; channel--; dest += channelSize)
        {
            // Iterate over the horizontal and vertical windowed regions
            for (int kernelRow = 0; kernelRow < windowHeight; kernelRow++)
            {
                for (int kernelCol = 0; kernelCol < windowWidth; kernelCol++)
                {
                    // Add the data inside this column to the appropriate cells
                    // in 'dest'.
                    int srcRow = -yPad + kernelRow * yDilation;
                    for (int destRows = srcHeight; destRows; destRows--)
                    {
                        if (!(srcRow >= 0 && srcRow < destHeight))
                            src += srcWidth;

                        else
                        {
                            int srcCol = -xPad + kernelCol * xDilation;
                            for (int destCol = srcWidth; destCol; destCol--)
                            {
                                if (srcCol >= 0 && srcCol < destWidth)
                                    dest[srcRow * destWidth + srcCol] += *src;

                                src++;
                                srcCol += xStride;
                            }
                        }
                        srcRow += yStride;
                    }
                }
            }
        }
    }

    static void im2Row(const T* src,
        const size_t srcWidth, const size_t srcHeight, const size_t channels,
        const size_t windowWidth, const size_t windowHeight,
        const size_t xPad, const size_t yPad,
        const size_t xStride, const size_t yStride, T* dest)
    {
        // Save some useful values
        const size_t NUM_HORIZONTAL_BLOCKS =
            ((srcWidth - windowWidth + 2*xPad) / xStride) + 1;
        const size_t NUM_VERTICAL_BLOCKS =
            ((srcHeight - windowHeight + 2*yPad) / yStride) + 1;
        const size_t OUT_WIDTH  = windowWidth * windowHeight * channels;
        //const size_t OUT_HEIGHT = NUM_HORIZONTAL_BLOCKS * NUM_VERTICAL_BLOCKS;

        for (size_t channel = 0; channel < channels; ++channel)
        {
            size_t destY = 0;

            // Iterate over each block in src
            int srcY = -yPad;
            for (size_t blockY = 0; blockY < NUM_VERTICAL_BLOCKS; ++blockY)
            {
                int srcX = -xPad;
                for (size_t blockX = 0; blockX < NUM_HORIZONTAL_BLOCKS; ++blockX)
                {
                    // Copy this block from src to dest
                    for (size_t dy = 0; dy < windowHeight; ++dy)
                    {
                        for (size_t dx = 0; dx < windowWidth; ++dx)
                        {
                            int x = srcX + dx;
                            int y = srcY + dy;

                            size_t destX = (dy * windowWidth + dx) +
                                (channel * (windowWidth * windowHeight));

                            if (x >= 0 && x < srcWidth && y >= 0 && y < srcHeight)
                                dest[destY * OUT_WIDTH + destX] = src[y * srcWidth + x];
                            else dest[destY * OUT_WIDTH + destX] = T{};
                        }
                    }

                    // Move forward one block
                    srcX += xStride;
                    ++destY;
                }
                srcY += yStride;
            }

            // Advance src to the next channel
            src += srcWidth * srcHeight;
        }
    }

    static void row2Im(const T* src,
        const size_t windowWidth, const size_t windowHeight, const size_t channels,
        const size_t destWidth, const size_t destHeight,
        const size_t xPad, const size_t yPad,
        const size_t xStride, const size_t yStride, T* dest)
    {
        // Save some useful values
        const size_t NUM_HORIZONTAL_BLOCKS =
            ((destWidth - windowWidth + 2*xPad) / xStride) + 1;
        const size_t NUM_VERTICAL_BLOCKS =
            ((destHeight - windowHeight + 2*yPad) / yStride) + 1;
        // const size_t SRC_ROWS  = NUM_HORIZONTAL_BLOCKS * NUM_VERTICAL_BLOCKS;
        const size_t SRC_COLS  = windowWidth * windowHeight * channels;

        // Ensure each element of 'dest' starts out at 0 so the accumulation logic
        // makes sense.
        std::fill(dest, dest + destWidth * destHeight * channels, T{});

        for (size_t c = 0; c < channels; ++c)
        {
            // Figure out which data in 'src' we're working with
            const T* srcStart = src + (c * windowWidth * windowHeight);

            // Iterate over each patch in the src matrix
            int destY = -yPad;
            for (size_t blockY = 0; blockY < NUM_VERTICAL_BLOCKS; ++blockY)
            {
                int destX = -xPad;
                for (size_t blockX = 0; blockX < NUM_HORIZONTAL_BLOCKS; ++blockX)
                {
                    // Add the contents of this patch to the corresponding cells
                    // in src
                    for (size_t dy = 0; dy < windowHeight; ++dy)
                    {
                        for (size_t dx = 0; dx < windowWidth; ++dx)
                        {
                            int x = destX + dx;
                            int y = destY + dy;

                            if (x >= 0 && x < destWidth &&
                                y >= 0 && y < destHeight)
                            {
                                int destIndex = destWidth * (y * channels + c) + x;
                                int srcIndex  = dy * windowWidth + dx;

                                dest[destIndex] += srcStart[srcIndex];
                            }
                        }
                    }

                    // Move to the next horizontal patch
                    destX    += xStride;
                    srcStart += SRC_COLS;
                }

                // Move to the next vertical patch
                destY += yStride;
            }
        }
    }
};

}

#endif

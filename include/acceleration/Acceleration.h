/*
 * File:   Acceleration.h
 * Author: Jon C. Hammer
 *
 * Created on December 11, 2016, 2:00 PM
 */

 #ifndef ACCELERATION_H
 #define ACCELERATION_H

#include "util/Assert.h"

// If user hasn't specified an acceleration framework, use CPU only as the default
#if !defined OPKIT_CPU_ONLY && !defined OPKIT_OPEN_BLAS && !defined OPKIT_NVBLAS
    #define OPKIT_CPU_ONLY
#endif

// Determine which implementation should be used
#ifdef OPKIT_CPU_ONLY
    #include "Acceleration_CPU.h"
    template <class T>
    using Accelerator = tensorlib::Acceleration_CPU<T>;
#endif

#ifdef OPKIT_OPEN_BLAS
    #include "Acceleration_OpenBlas.h"
    template <class T>
    using Accelerator = tensorlib::Acceleration_OpenBlas<T>;
#endif

#ifdef OPKIT_NVBLAS
    #include "Acceleration_NVBlas.h"
    template <class T>
    using Accelerator = tensorlib::Acceleration_NVBlas<T>;
#endif

namespace detail
{
    // Default matrix multiplication routine. Works for arbitrary x and y strides
    // for all three matrices. BLAS routines only support an arbitrary stride
    // in the y-dimension.
    template <class T>
    void defaultMMMultiply(const T* A, const T* B, T* C,
        const size_t M, const size_t N, const size_t K,
        const size_t aStrideX, const size_t aStrideY,
        const size_t bStrideX, const size_t bStrideY,
        const size_t cStrideX, const size_t cStrideY,
        const T alpha = T{1}, const T beta = T{})
    {
        //std::cerr << "WARNING: DEFAULT MATRIX MULTIPLY USED." << std::endl;

        size_t aY = 0;
        size_t cY = 0;

        for (size_t i = 0; i < M; ++i)
        {
            size_t bX = 0;
            size_t cX = 0;

            for (size_t j = 0; j < N; ++j)
            {
                size_t aX = 0;
                size_t bY = 0;

                T sum = T{};
                for (size_t k = 0; k < K; ++k)
                {
                    sum += A[aY + aX] * B[bY + bX];
                    aX  += aStrideX;
                    bY  += bStrideY;
                }

                C[cY + cX] *= beta;
                C[cY + cX] += alpha * sum;

                bX += bStrideX;
                cX += cStrideX;
            }

            aY += aStrideY;
            cY += cStrideY;
        }
    }

    // Default matrix multiplication routine. Works for arbitrary x and y strides
    // for all three matrices. BLAS routines only support an arbitrary stride
    // in the y-dimension.
    template <class T>
    void defaultMMTMultiply(const T* A, const T* B, T* C,
        const size_t M, const size_t N, const size_t K,
        const size_t aStrideX, const size_t aStrideY,
        const size_t bStrideX, const size_t bStrideY,
        const size_t cStrideX, const size_t cStrideY,
        const T alpha = T{1}, const T beta = T{})
    {
        //std::cerr << "WARNING: DEFAULT MATRIX MULTIPLY USED." << std::endl;

        size_t aY = 0;
        size_t cY = 0;

        for (size_t i = 0; i < M; ++i)
        {
            size_t bY = 0;
            size_t cX = 0;

            for (size_t j = 0; j < N; ++j)
            {
                size_t aX = 0;
                size_t bX = 0;

                T sum = T{};
                for (size_t k = 0; k < K; ++k)
                {
                    sum += A[aY + aX] * B[bY + bX];
                    aX  += aStrideX;
                    bX  += bStrideX;
                }

                C[cY + cX] *= beta;
                C[cY + cX] += alpha * sum;

                bY += bStrideY;
                cX += cStrideX;
            }

            aY += aStrideY;
            cY += cStrideY;
        }
    }

    // Default matrix multiplication routine. Works for arbitrary x and y strides
    // for all three matrices. BLAS routines only support an arbitrary stride
    // in the y-dimension.
    template <class T>
    void defaultMTMMultiply(const T* A, const T* B, T* C,
        const size_t M, const size_t N, const size_t K,
        const size_t aStrideX, const size_t aStrideY,
        const size_t bStrideX, const size_t bStrideY,
        const size_t cStrideX, const size_t cStrideY,
        const T alpha = T{1}, const T beta = T{})
    {
        //std::cerr << "WARNING: DEFAULT MATRIX MULTIPLY USED." << std::endl;

        size_t aX = 0;
        size_t cY = 0;

        for (size_t i = 0; i < M; ++i)
        {
            size_t bX = 0;
            size_t cX = 0;

            for (size_t j = 0; j < N; ++j)
            {
                size_t aY = 0;
                size_t bY = 0;

                T sum = T{};
                for (size_t k = 0; k < K; ++k)
                {
                    sum += A[aY + aX] * B[bY + bX];
                    aY  += aStrideY;
                    bY  += bStrideY;
                }

                C[cY + cX] *= beta;
                C[cY + cX] += alpha * sum;

                bX += bStrideX;
                cX += cStrideX;
            }

            aX += aStrideX;
            cY += cStrideY;
        }
    }
}

namespace tensorlib
{
    // Computes C = alpha * A * B + beta * C, where A is an M x K
    // matrix, B is a K x N matrix, C is an M x N matrix, alpha is
    // a scalar, and beta is a scalar.
    template <class T>
    inline void mmMultiply(const T* A, const T* B, T* C,
        const size_t M, const size_t N, const size_t K,
        const size_t aStrideX, const size_t aStrideY,
        const size_t bStrideX, const size_t bStrideY,
        const size_t cStrideX, const size_t cStrideY,
        const T alpha = T{1}, const T beta = T{})
    {
        if (aStrideX == 1 && bStrideX == 1 && cStrideX == 1 &&
            aStrideY != 0 && bStrideY != 0 && cStrideY != 0)
            Accelerator<T>::mmMultiply(A, B, C, M, N, K, aStrideY, bStrideY, cStrideY, alpha, beta);
        else detail::defaultMMMultiply(A, B, C, M, N, K, aStrideX, aStrideY,
            bStrideX, bStrideY, cStrideX, cStrideY, alpha, beta);
    }

    // Computes C = alpha * A * B^T + beta * C, where A is an M x K
    // matrix, B is an N x K matrix, C is an M x N matrix, alpha is
    // a scalar, and beta is a scalar.
    template <class T>
    inline void mmtMultiply(const T* A, const T* B, T* C,
        const size_t M, const size_t N, const size_t K,
        const size_t aStrideX, const size_t aStrideY,
        const size_t bStrideX, const size_t bStrideY,
        const size_t cStrideX, const size_t cStrideY,
        const T alpha = T{1.0}, const T beta = T{0.0})
    {
        if (aStrideX == 1 && bStrideX == 1 && cStrideX == 1 &&
            aStrideY != 0 && bStrideY != 0 && cStrideY != 0)
            Accelerator<T>::mmtMultiply(A, B, C, M, N, K, aStrideY, bStrideY, cStrideY, alpha, beta);
        else detail::defaultMMTMultiply(A, B, C, M, N, K, aStrideX, aStrideY,
            bStrideX, bStrideY, cStrideX, cStrideY, alpha, beta);
    }

    // Computes C = alpha * A^T * B + beta * C, where A is a K x M
    // matrix, B is a K x N matrix, C is an M x N matrix, alpha is
    // a scalar, and beta is a scalar.
    template <class T>
    inline void mtmMultiply(const T* A, const T* B, T* C,
        const size_t M, const size_t N, const size_t K,
        const size_t aStrideX, const size_t aStrideY,
        const size_t bStrideX, const size_t bStrideY,
        const size_t cStrideX, const size_t cStrideY,
        const T alpha = T{1.0}, const T beta = T{0.0})
    {
        if (aStrideX == 1 && bStrideX == 1 && cStrideX == 1 &&
            aStrideY != 0 && bStrideY != 0 && cStrideY != 0)
            Accelerator<T>::mtmMultiply(A, B, C, M, N, K, aStrideY, bStrideY, cStrideY, alpha, beta);
        else detail::defaultMTMMultiply(A, B, C, M, N, K, aStrideX, aStrideY,
            bStrideX, bStrideY, cStrideX, cStrideY, alpha, beta);
    }

    // Computes y = alpha * A * x + beta * y, where A is an M x N
    // matrix, x is a vector of size N, y is a vector of size M,
    // and alpha and beta are scalars.
    // xInc and yInc can be adjusted if the vectors are stored
    // in an interlaced format.
    template <class T>
    inline void mvMultiply(const T* A, const T* x, T* y,
        const size_t M, const size_t N,
        const T alpha = T{1.0}, const T beta = T{0.0},
        const int xInc = 1, const int yInc = 1)
    {
        Accelerator<T>::mvMultiply(A, x, y, M, N, alpha, beta, xInc, yInc);
    }

    // Computes y = alpha * A^T * x + beta * y, where A is an M x N
    // matrix, x is a vector of size M, y is a vector of size N,
    // and alpha and beta are scalars.
    // xInc and yInc can be adjusted if the vectors are stored
    // in an interlaced format.
    template <class T>
    inline void mtvMultiply(const T* A, const T* x, T* y,
        const size_t M, const size_t N,
        const T alpha = T{1.0}, const T beta = T{0.0},
        const int xInc = 1, const int yInc = 1)
    {
        Accelerator<T>::mtvMultiply(A, x, y, M, N, alpha, beta, xInc, yInc);
    }

    // Computes y = alpha * A * x + beta * y, where A is an N x N
    // symmetric matrix, x is a vector of size N, y is a vector of size N,
    // and alpha and beta are scalars.
    // xInc and yInc can be adjusted if the vectors are stored
    // in an interlaced format.
    template <class T>
    inline void symmetricMvMultiply(const T* A, const T* x, T* y,
        const size_t N, const T alpha = T{1.0}, const T beta = {0.0},
        const int xInc = 1, const int yInc = 1)
    {
        Accelerator<T>::symmetricMvMultiply(A, x, y, N, alpha, beta, xInc, yInc);
    }

    // Adds alpha * x * y^T to A, where x is a vector of size M,
    // y is a vector of size N, A is a M x N matrix, and alpha is
    // a scalar. When A is initialized to 0's, this calculates the
    // vector outer product between x and y. Otherwise, it performs
    // a rank-1 update of A.
    // xInc and yInc can be adjusted if the vectors are stored
    // in an interlaced format.
    template <class T>
    inline void outerProduct(const T* x, const T* y, T* A,
        const size_t M, const size_t N, const T alpha = T{1.0},
        const int xInc = 1, const int yInc = 1)
    {
        Accelerator<T>::outerProduct(x, y, A, M, N, alpha, xInc, yInc);
    }

    // Computes y += alpha * x, where x is a vector of size N,
    // y is a vector of size N, and alpha is a scalar.
    // xInc and yInc can be adjusted if the vectors are stored
    // in an interlaced format.
    template <class T>
    inline void vAdd(const T* x, T* y,
        const size_t N, const T alpha = T{1.0},
        const int xInc = 1, const int yInc = 1)
    {
        Accelerator<T>::vAdd(x, y, N, alpha, xInc, yInc);
    }

    // Computes x = alpha * x, where x is a vector of size N and
    // alpha is a scalar. xInc can be adjusted if the vector is
    // stored in an interlaced format.
    template <class T>
    inline void vScale(T* x, const T alpha, const size_t N, const int xInc = 1)
    {
        Accelerator<T>::vScale(x, alpha, N, xInc);
    }

    // Returns the index where the maximum element is found in the vector x of
    // size N. xInc can be adjusted if the vector is
    // stored in an interlaced format.
    template <class T>
    inline size_t vMaxIndex(const T* x, const size_t N, const int xInc = 1)
    {
        return Accelerator<T>::vMaxIndex(x, N, xInc);
    }

    // Copies the contents of x into y, where x and y are vectors of size N.
    // xInc and yInc can be adjusted if the vectors are stored in an
    // interlaced format.
    template <class T>
    inline void vCopy(const T* x, T* y, const size_t N,
        const int xInc = 1, const int yInc = 1)
    {
        Accelerator<T>::vCopy(x, y, N, xInc, yInc);
    }

    // C = alpha * A + beta * C
    // Both matrices are M x N. Each matrix may have an arbitrary stride in both
    // the x and y directions.
    template <class T>
    inline void matrixAdd(const T* A, T* C,
        const size_t M, const size_t N,
        const size_t aStrideX, const size_t aStrideY,
        const size_t cStrideX, const size_t cStrideY,
        const T alpha, const T beta)
    {
        Accelerator<T>::matrixAdd(A, C, M, N, aStrideX, aStrideY,
            cStrideX, cStrideY, alpha, beta);
    }

    // y = alpha * x + beta * y
    // Both vectors are of length N.
    template <class T>
    inline void axpby(const T* x, T* y, const size_t N, const T alpha,
        const T beta, const size_t xInc, const size_t yInc)
    {
        Accelerator<T>::axpby(x, y, N, alpha, beta, xInc, yInc);
    }

    // Input  - srcWidth x srcHeight x channels image. Channels stored consecutively.
    // Output - (windowWidth * windowHeight) x (numPatches) x channels image.
    //          Channels stored consecutively.
    template <class T>
    void im2col(const T* src,
        const size_t srcWidth, const size_t srcHeight, const size_t channels,
        const size_t windowWidth, const size_t windowHeight,
        const size_t xPad, const size_t yPad,
        const size_t xStride, const size_t yStride,
        const size_t xDilation, const size_t yDilation, T* dest)
    {
        Accelerator<T>::im2col(src, srcWidth, srcHeight, channels,
            windowWidth, windowHeight, xPad, yPad, xStride, yStride,
            xDilation, yDilation, dest);
    }

    // Input  - (windowWidth * windowHeight) x (numPatches) x channels image.
    //          Channels stored consecutively.
    // Output - srcWidth x srcHeight x channels image. Channels stored consecutively.
    template <class T>
    void col2im(const T* src,
        const int destWidth, const int destHeight, const int channels,
        const int windowWidth, const int windowHeight,
        const int xPad, const int yPad,
        const int xStride, const int yStride,
        const int xDilation, const int yDilation,
        T* dest)
    {
        Accelerator<T>::col2im(src, destWidth, destHeight, channels,
            windowWidth, windowHeight, xPad, yPad, xStride, yStride,
            xDilation, yDilation, dest);
    }

    // This function is similar to the im2Col function found in Matlab. Given a
    // source tensor of dimensions (srcWidth * srcHeight * channels), this function
    // isolates each (windowWidth * windowHeight * channels) patch and copies it
    // into the given destination matrix as a single row.
    //
    // The destination matrix will have dimensions (K * N), where:
    // K = The number of patches = NumHorizontalBlocks * NumVerticalBlocks, where:
    //     NumHorizontalBlocks = ((srcWidth - windowWidth + 2*xPad) / xStride) + 1
    //     NumVerticalBlocks = ((srcHeight - windowHeight + 2*yPad) / yStride) + 1
    // N = windowWidth * windowHeight * channels
    //
    // 'src' is assumed to be in row-major order. Channels are stored sequentially,
    // rather than interleaved. 'dest' will also be filled in row-major order.
    //
    // The four remaining parameters determine how the window will slide across the
    // source tensor. 'xPad' and 'yPad' determine the amount of zero-padding to
    // apply in each dimension. 'xStride' and 'yStride' determine the window stride.
    // A larger stride will result in a smaller result, since some of the input
    // cells will be skipped over.
    template <class T>
    void im2Row(const T* src,
        const size_t srcWidth, const size_t srcHeight, const size_t channels,
        const size_t windowWidth, const size_t windowHeight,
        const size_t xPad, const size_t yPad,
        const size_t xStride, const size_t yStride, T* dest)
    {
        Accelerator<T>::im2Row(src, srcWidth, srcHeight, channels,
            windowWidth, windowHeight, xPad, yPad, xStride, yStride, dest);
    }

    // This function can be thought of as an inverse to im2Row. Given a source
    // matrix with dimensions (K * N) x (windowWidth * windowHeight * channels),
    // where:
    // K = The number of patches = NumHorizontalBlocks * NumVerticalBlocks, where:
    //     NumHorizontalBlocks = ((srcWidth - windowWidth + 2*xPad) / xStride) + 1
    //     NumVerticalBlocks = ((destHeight - windowHeight + 2*yPad) / yStride) + 1
    // N = windowWidth * windowHeight * channels
    //
    // Each row of the source is interpreted as a single patch in an original 2D
    // image with dimensions (destWidth x destHeight x channels). This function will
    // add the contributions from each patch to essentially reconstruct the original
    // image. Note that the reconstruction will not be exact, unless the stride
    // matches the window dimensions. Otherwise, many elements will be counted
    // repeatedly.
    //
    // 'src' is assumed to be in row-major order. Channels are stored sequentially,
    // rather than interleaved. 'dest' will also be filled in row-major order.
    //
    // The four remaining parameters determine how the window will slide across the
    // source tensor. 'xPad' and 'yPad' determine the amount of zero-padding to
    // apply in each dimension. 'xStride' and 'yStride' determine the window stride.
    // A larger stride will result in a smaller result, since some of the input
    // cells will be skipped over. These should be the same parameters that were
    // used when im2Row was originally called.
    template <class T>
    void row2Im(const T* src,
        const size_t windowWidth, const size_t windowHeight, const size_t channels,
        const size_t destWidth, const size_t destHeight,
        const size_t xPad, const size_t yPad,
        const size_t xStride, const size_t yStride, T* dest)
    {
        Accelerator<T>::row2Im(src, windowWidth, windowHeight, channels,
            destWidth, destHeight, xPad, yPad, xStride, yStride, dest);
    }
}

#endif

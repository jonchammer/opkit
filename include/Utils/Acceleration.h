/*
 * File:   Acceleration.h
 * Author: Jon C. Hammer
 *
 * Created on December 11, 2016, 2:00 PM
 */

 #ifndef ACCELERATION_H
 #define ACCELERATION_H

namespace opkit
{
    // Determine which implementation should be used
    #ifdef OPKIT_CPU_ONLY
        #include "Acceleration_CPU.h"
        template <class T>
        using Accelerator = Acceleration<T>;
    #endif

    #ifdef OPKIT_OPEN_BLAS
        #include "Acceleration_OpenBlas.h"
        template <class T>
        using Accelerator = OpenBlas_Acceleration<T>;
    #endif

    #ifdef OPKIT_NVBLAS
        #include "Acceleration_NVBlas.h"
        template <class T>
        using Accelerator = NVBlas_Acceleration<T>;
    #endif

    // Computes C = alpha * A * B + beta * C, where A is an M x K
    // matrix, B is a K x N matrix, C is an M x N matrix, alpha is
    // a scalar, and beta is a scalar.
    template <class T>
    inline void mmMultiply(const T* A, const T* B, T* C,
        const size_t M, const size_t N, const size_t K,
        const T alpha = T{1.0}, const T beta = T{0.0})
    {
        Accelerator<T>::mmMultiply(A, B, C, M, N, K, alpha, beta);
    }

    // Computes C = alpha * A * B^T + beta * C, where A is an M x K
    // matrix, B is an N x K matrix, C is an M x N matrix, alpha is
    // a scalar, and beta is a scalar.
    template <class T>
    inline void mmtMultiply(const T* A, const T* B, T* C,
        const size_t M, const size_t N, const size_t K,
        const T alpha = T{1.0}, const T beta = T{0.0})
    {
        Accelerator<T>::mmtMultiply(A, B, C, M, N, K, alpha, beta);
    }

    // Computes C = alpha * A^T * B + beta * C, where A is a K x M
    // matrix, B is a K x N matrix, C is an M x N matrix, alpha is
    // a scalar, and beta is a scalar.
    template <class T>
    inline void mtmMultiply(const T* A, const T* B, T* C,
        const size_t M, const size_t N, const size_t K,
        const T alpha = T{1.0}, const T beta = T{0.0})
    {
        Accelerator<T>::mtmMultiply(A, B, C, M, N, K, alpha, beta);
    }

    // Computes C = alpha * A * B + beta * C, where A, B, and C are multi-
    // channel matrices (or 3rd order tensors). Each channel is processed
    // separately. A is assumed to be an M x K matrix containing 'numChannels'
    // channels. Similarly, B is a K x N x numChannels tensor, and the result,
    // C, is an M x N x numChannels tensor. Each MxK, KxN, or MxN elements
    // constitute a single channel (for A, B, and C, respectively).
    template <class T>
    inline void channeledMMMultiply(const T* A, const T* B, T* C,
        const size_t M, const size_t N, const size_t K, const size_t numChannels,
        const T alpha = T{1.0}, const T beta = T{0.0})
    {
        Accelerator<T>::channeledMMMultiply(A, B, C, M, N, K, numChannels, alpha, beta);
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
        Accelerator<T>::vMaxIndex(x, N, xInc);
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















// #include <cblas.h>
//
// #ifdef OPKIT_NVBLAS
//
// #endif
//
// namespace opkit
// {
//
// // OpenBlas translation macros. These allow us to convert from one BLAS library's
// // syntax to another.
// #ifdef OPENBLAS_CONFIG_H
//
//     #define OPKIT_DGEMM(transposeA, transposeB,                                \
//         M, N, K,                                                               \
//         alpha, A, ldA,                                                         \
//         B, ldB, beta,                                                          \
//         C, ldC)                                                                \
//         cblas_dgemm(CblasColMajor, transposeA, transposeB, M, N, K, alpha, A,  \
//             ldA, B, ldB, beta, C, ldC);                                        \
//
//     #define OPKIT_SGEMM(transposeA, transposeB,                                \
//         M, N, K,                                                               \
//         alpha, A, ldA,                                                         \
//         B, ldB, beta,                                                          \
//         C, ldC)                                                                \
//         cblas_sgemm(CblasColMajor, transposeA, transposeB, M, N, K, alpha, A,  \
//             ldA, B, ldB, beta, C, ldC);                                        \
//
//     #define OPKIT_DGEMV(transposeA,                                            \
//         M, N,                                                                  \
//         alpha, A, ldA,                                                         \
//         x, incX, beta,                                                         \
//         y, incY)                                                               \
//         cblas_dgemv(CblasColMajor, transposeA, M, N, alpha, A, ldA, x, incX,   \
//             beta, y, incY);                                                    \
//
//     #define OPKIT_SGEMV(transposeA,                                            \
//         M, N,                                                                  \
//         alpha, A, ldA,                                                         \
//         x, incX, beta,                                                         \
//         y, incY)                                                               \
//         cblas_sgemv(CblasColMajor, transposeA, M, N, alpha, A, ldA, x, incX,   \
//             beta, y, incY);                                                    \
//
//     #define OPKIT_DSYMV(upper, N, alpha, A, ldA, x, incX, beta, y, incY)       \
//         cblas_dsymv(CblasColMajor, upper, N, alpha, A, ldA,                    \
//             x, incX, beta, y, incY);                                           \
//
//     #define OPKIT_SSYMV(upper, N, alpha, A, ldA, x, incX, beta, y, incY)       \
//         cblas_ssymv(CblasColMajor, upper, N, alpha, A, ldA,                    \
//             x, incX, beta, y, incY);                                           \
//
//     #define OPKIT_DGER(M, N, alpha, x, incX, y, incY, A, ldA)                  \
//         cblas_dger(CblasColMajor, M, N, alpha, x, incX, y, incY, A, ldA);      \
//
//     #define OPKIT_SGER(M, N, alpha, x, incX, y, incY, A, ldA)                  \
//         cblas_sger(CblasColMajor, M, N, alpha, x, incX, y, incY, A, ldA);      \
//
//     #define OPKIT_DAXPY(N, alpha, x, incX, y, incY)                            \
//         cblas_daxpy(N, alpha, x, incX, y, incY);                               \
//
//     #define OPKIT_SAXPY(N, alpha, x, incX, y, incY)                            \
//         cblas_saxpy(N, alpha, x, incX, y, incY);                               \
//
//     #define OPKIT_DSCAL(N, alpha, X, incX) cblas_dscal(N, alpha, X, incX);
//     #define OPKIT_SSCAL(N, alpha, X, incX) cblas_sscal(N, alpha, X, incX);
//
//     #define OPKIT_IDAMAX(N, x, incX) cblas_idamax(N, x, incX);
//     #define OPKIT_ISAMAX(N, x, incX) cblas_isamax(N, x, incX);
//
//     #define OPKIT_DCOPY(N, x, incX, y, incY) cblas_dcopy(N, x, incX, y, incY);
//     #define OPKIT_SCOPY(N, x, incX, y, incY) cblas_scopy(N, x, incX, y, incY);
//
//     // BLAS parameter macros
//     #define OPKIT_L3_TRANSPOSE CblasTrans
//     #define OPKIT_L3_NO_TRANSPOSE CblasNoTrans
//     #define OPKIT_L2_TRANSPOSE CblasTrans
//     #define OPKIT_L2_NO_TRANSPOSE CblasNoTrans
//     #define OPKIT_UPPER CblasUpper
//
//
// #endif
//
// // NVBlas translation macros
// // This symbol will be set by the user to indicate that they desire to use a
// // GPU to accelerate the L3 BLAS operations. The lower level operations will
// // have to come from another CPU BLAS library.
// #ifdef OPKIT_NVBLAS
//
//     #undef OPKIT_DGEMM
//     #undef OPKIT_SGEMM
//     #undef OPKIT_L3_TRANSPOSE
//     #undef OPKIT_L3_NO_TRANSPOSE
//
//     #define OPKIT_DGEMM(transposeA, transposeB,                                \
//         M, N, K,                                                               \
//         alpha, A, ldA,                                                         \
//         B, ldB, beta,                                                          \
//         C, ldC)                                                                \
//     {                                                                          \
//         int m   = (int) M;                                                     \
//         int n   = (int) N;                                                     \
//         int k   = (int) K;                                                     \
//         int lda = (int) ldA;                                                   \
//         int ldb = (int) ldB;                                                   \
//         int ldc = (int) ldC;                                                   \
//         dgemm(transposeA, transposeB, &m, &n, &k, &alpha, A,                   \
//             &lda, B, &ldb, &beta, C, &ldc);                                    \
//     }                                                                          \
//
//     #define OPKIT_SGEMM(transposeA, transposeB,                                \
//         M, N, K,                                                               \
//         alpha, A, ldA,                                                         \
//         B, ldB, beta,                                                          \
//         C, ldC)                                                                \
//     {                                                                          \
//         int m   = (int) M;                                                     \
//         int n   = (int) N;                                                     \
//         int k   = (int) K;                                                     \
//         int lda = (int) ldA;                                                   \
//         int ldb = (int) ldB;                                                   \
//         int ldc = (int) ldC;                                                   \
//         sgemm(transposeA, transposeB, &m, &n, &k, &alpha, A,                   \
//             &lda, B, &ldb, &beta, C, &ldc);                                    \
//     }                                                                          \
//
//     #define OPKIT_L3_TRANSPOSE "T"
//     #define OPKIT_L3_NO_TRANSPOSE "N"
// #endif
//
// // For BLAS reference see:
// // https://software.intel.com/sites/default/files/managed/ff/c8/mkl-2017-developer-reference-c_0.pdf
// //
// // NOTE: Some BLAS libraries make the assumption that matrices are stored
// // in column-major order. Since our data is actually in row-major order,
// // some sort of conversion would normally have to be applied in order to use
// // those libraries. Helpfully, it is usually possible to perform an
// // alternate computation (e.g. by switching the dimensions and positions of
// // operands) to trick the library into doing the same work, despite the
// // differences in ordering. The accelerated routines that use matrices below
// // make use of these tricks so the underlying BLAS library always believes
// // it is working on data in column-major order. This makes it easier to
// // switch between libraries, since some (e.g. OpenBLAS) do these conversions
// // automatically, but some (e.g. NVBlas) do not.
// };


 #endif

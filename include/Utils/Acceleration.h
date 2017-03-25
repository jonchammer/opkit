/*
 * File:   Acceleration.h
 * Author: Jon C. Hammer
 *
 * Created on December 11, 2016, 2:00 PM
 */

 #ifndef ACCELERATION_H
 #define ACCELERATION_H

#include <cblas.h>

#ifdef OPKIT_NVBLAS
    #include <nvblas.h>
#endif

namespace opkit
{

// OpenBlas translation macros. These allow us to convert from one BLAS library's
// syntax to another.
#ifdef OPENBLAS_CONFIG_H

    #define OPKIT_DGEMM(transposeA, transposeB,                                \
        M, N, K,                                                               \
        alpha, A, ldA,                                                         \
        B, ldB, beta,                                                          \
        C, ldC)                                                                \
        cblas_dgemm(CblasColMajor, transposeA, transposeB, M, N, K, alpha, A,  \
            ldA, B, ldB, beta, C, ldC);                                        \

    #define OPKIT_SGEMM(transposeA, transposeB,                                \
        M, N, K,                                                               \
        alpha, A, ldA,                                                         \
        B, ldB, beta,                                                          \
        C, ldC)                                                                \
        cblas_sgemm(CblasColMajor, transposeA, transposeB, M, N, K, alpha, A,  \
            ldA, B, ldB, beta, C, ldC);                                        \

    #define OPKIT_DGEMV(transposeA,                                            \
        M, N,                                                                  \
        alpha, A, ldA,                                                         \
        x, incX, beta,                                                         \
        y, incY)                                                               \
        cblas_dgemv(CblasColMajor, transposeA, M, N, alpha, A, ldA, x, incX,   \
            beta, y, incY);                                                    \

    #define OPKIT_SGEMV(transposeA,                                            \
        M, N,                                                                  \
        alpha, A, ldA,                                                         \
        x, incX, beta,                                                         \
        y, incY)                                                               \
        cblas_sgemv(CblasColMajor, transposeA, M, N, alpha, A, ldA, x, incX,   \
            beta, y, incY);                                                    \

    #define OPKIT_DSYMV(upper, N, alpha, A, ldA, x, incX, beta, y, incY)       \
        cblas_dsymv(CblasColMajor, upper, N, alpha, A, ldA,                    \
            x, incX, beta, y, incY);                                           \

    #define OPKIT_SSYMV(upper, N, alpha, A, ldA, x, incX, beta, y, incY)       \
        cblas_ssymv(CblasColMajor, upper, N, alpha, A, ldA,                    \
            x, incX, beta, y, incY);                                           \

    #define OPKIT_DGER(M, N, alpha, x, incX, y, incY, A, ldA)                  \
        cblas_dger(CblasColMajor, M, N, alpha, x, incX, y, incY, A, ldA);      \

    #define OPKIT_SGER(M, N, alpha, x, incX, y, incY, A, ldA)                  \
        cblas_sger(CblasColMajor, M, N, alpha, x, incX, y, incY, A, ldA);      \

    #define OPKIT_DAXPY(N, alpha, x, incX, y, incY)                            \
        cblas_daxpy(N, alpha, x, incX, y, incY);                               \

    #define OPKIT_SAXPY(N, alpha, x, incX, y, incY)                            \
        cblas_saxpy(N, alpha, x, incX, y, incY);                               \

    #define OPKIT_DSCAL(N, alpha, X, incX) cblas_dscal(N, alpha, X, incX);
    #define OPKIT_SSCAL(N, alpha, X, incX) cblas_sscal(N, alpha, X, incX);

    #define OPKIT_IDAMAX(N, x, incX) cblas_idamax(N, x, incX);
    #define OPKIT_ISAMAX(N, x, incX) cblas_isamax(N, x, incX);

    #define OPKIT_DCOPY(N, x, incX, y, incY) cblas_dcopy(N, x, incX, y, incY);
    #define OPKIT_SCOPY(N, x, incX, y, incY) cblas_scopy(N, x, incX, y, incY);

    // BLAS parameter macros
    #define OPKIT_L3_TRANSPOSE CblasTrans
    #define OPKIT_L3_NO_TRANSPOSE CblasNoTrans
    #define OPKIT_L2_TRANSPOSE CblasTrans
    #define OPKIT_L2_NO_TRANSPOSE CblasNoTrans
    #define OPKIT_UPPER CblasUpper

    #define USE_ALL_CORES() openblas_set_num_threads(openblas_get_num_procs())
    #define USE_ONE_CORE()  openblas_set_num_threads(1)
#endif

// NVBlas translation macros
// This symbol will be set by the user to indicate that they desire to use a
// GPU to accelerate the L3 BLAS operations. The lower level operations will
// have to come from another CPU BLAS library.
#ifdef OPKIT_NVBLAS

    #undef OPKIT_DGEMM
    #undef OPKIT_SGEMM
    #undef OPKIT_L3_TRANSPOSE
    #undef OPKIT_L3_NO_TRANSPOSE

    #define OPKIT_DGEMM(transposeA, transposeB,                                \
        M, N, K,                                                               \
        alpha, A, ldA,                                                         \
        B, ldB, beta,                                                          \
        C, ldC)                                                                \
    {                                                                          \
        int m   = (int) M;                                                     \
        int n   = (int) N;                                                     \
        int k   = (int) K;                                                     \
        int lda = (int) ldA;                                                   \
        int ldb = (int) ldB;                                                   \
        int ldc = (int) ldC;                                                   \
        dgemm(transposeA, transposeB, &m, &n, &k, &alpha, A,                   \
            &lda, B, &ldb, &beta, C, &ldc);                                    \
    }                                                                          \

    #define OPKIT_SGEMM(transposeA, transposeB,                                \
        M, N, K,                                                               \
        alpha, A, ldA,                                                         \
        B, ldB, beta,                                                          \
        C, ldC)                                                                \
    {                                                                          \
        int m   = (int) M;                                                     \
        int n   = (int) N;                                                     \
        int k   = (int) K;                                                     \
        int lda = (int) ldA;                                                   \
        int ldb = (int) ldB;                                                   \
        int ldc = (int) ldC;                                                   \
        sgemm(transposeA, transposeB, &m, &n, &k, &alpha, A,                   \
            &lda, B, &ldb, &beta, C, &ldc);                                    \
    }                                                                          \

    #define OPKIT_L3_TRANSPOSE "T"
    #define OPKIT_L3_NO_TRANSPOSE "N"
#endif

// For BLAS reference see:
// https://software.intel.com/sites/default/files/managed/ff/c8/mkl-2017-developer-reference-c_0.pdf
//
// NOTE: Some BLAS libraries make the assumption that matrices are stored
// in column-major order. Since our data is actually in row-major order,
// some sort of conversion would normally have to be applied in order to use
// those libraries. Helpfully, it is usually possible to perform an
// alternate computation (e.g. by switching the dimensions and positions of
// operands) to trick the library into doing the same work, despite the
// differences in ordering. The accelerated routines that use matrices below
// make use of these tricks so the underlying BLAS library always believes
// it is working on data in column-major order. This makes it easier to
// switch between libraries, since some (e.g. OpenBLAS) do these conversions
// automatically, but some (e.g. NVBlas) do not.

// Computes C = alpha * A * B + beta * C, where A is an M x K
// matrix, B is a K x N matrix, C is an M x N matrix, alpha is
// a scalar, and beta is a scalar.
inline void mmMultiply(const double* A, const double* B, double* C,
    const size_t M, const size_t N, const size_t K,
    const double alpha = 1.0, const double beta = 0.0)
{
    USE_ONE_CORE();
    //USE_ALL_CORES();

    OPKIT_DGEMM(OPKIT_L3_NO_TRANSPOSE, OPKIT_L3_NO_TRANSPOSE,
        N, M, K,
        alpha, B, N,
        A, K,
        beta, C, N);

    // Equivalently,
    // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    //     M, N, K,
    //     alpha, A, K,
    //     B, N,
    //     beta, C, N);
}

// Computes C = alpha * A * B + beta * C, where A is an M x K
// matrix, B is a K x N matrix, C is an M x N matrix, alpha is
// a scalar, and beta is a scalar.
inline void mmMultiply(const float* A, const float* B, float* C,
    const size_t M, const size_t N, const size_t K,
    const float alpha = 1.0f, const float beta = 0.0f)
{
    USE_ONE_CORE();
    //USE_ALL_CORES();

    OPKIT_SGEMM(OPKIT_L3_NO_TRANSPOSE, OPKIT_L3_NO_TRANSPOSE,
        N, M, K,
        alpha, B, N,
        A, K,
        beta, C, N);

    // Equivalently,
    // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    //     M, N, K,
    //     alpha, A, K,
    //     B, N,
    //     beta, C, N);
}

// Computes C = alpha * A * B^T + beta * C, where A is an M x K
// matrix, B is an N x K matrix, C is an M x N matrix, alpha is
// a scalar, and beta is a scalar.
inline void mmtMultiply(const double* A, const double* B, double* C,
    const size_t M, const size_t N, const size_t K,
    const double alpha = 1.0, const double beta = 0.0)
{
    USE_ONE_CORE();
    //USE_ALL_CORES();

    OPKIT_DGEMM(OPKIT_L3_TRANSPOSE, OPKIT_L3_NO_TRANSPOSE,
        N, M, K,
        alpha, B, K,
        A, K,
        beta, C, N);

    // Equivalently,
    // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
    //     M, N, K,
    //     alpha, A, K,
    //     B, K,
    //     beta, C, N);
}

// Computes C = alpha * A * B^T + beta * C, where A is an M x K
// matrix, B is an N x K matrix, C is an M x N matrix, alpha is
// a scalar, and beta is a scalar.
inline void mmtMultiply(const float* A, const float* B, float* C,
    const size_t M, const size_t N, const size_t K,
    const float alpha = 1.0f, const float beta = 0.0f)
{
    USE_ONE_CORE();
    //USE_ALL_CORES();

    OPKIT_SGEMM(OPKIT_L3_TRANSPOSE, OPKIT_L3_NO_TRANSPOSE,
        N, M, K,
        alpha, B, K,
        A, K,
        beta, C, N);

    // Equivalently,
    // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
    //     M, N, K,
    //     alpha, A, K,
    //     B, K,
    //     beta, C, N);
}

// Computes C = alpha * A^T * B + beta * C, where A is a K x M
// matrix, B is a K x N matrix, C is an M x N matrix, alpha is
// a scalar, and beta is a scalar.
inline void mtmMultiply(const double* A, const double* B, double* C,
    const size_t M, const size_t N, const size_t K,
    const double alpha = 1.0, const double beta = 0.0)
{
    USE_ONE_CORE();
    //USE_ALL_CORES();

    OPKIT_DGEMM(OPKIT_L3_NO_TRANSPOSE, OPKIT_L3_TRANSPOSE,
        N, M, K,
        alpha, B, N,
        A, M,
        beta, C, N);

    // Equivalently,
    // cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
    //     M, N, K,
    //     alpha, A, M,
    //     B, N,
    //     beta, C, N);
}

// Computes C = alpha * A^T * B + beta * C, where A is a K x M
// matrix, B is a K x N matrix, C is an M x N matrix, alpha is
// a scalar, and beta is a scalar.
inline void mtmMultiply(const float* A, const float* B, float* C,
    const size_t M, const size_t N, const size_t K,
    const float alpha = 1.0f, const float beta = 0.0f)
{
    USE_ONE_CORE();
    //USE_ALL_CORES();

    OPKIT_SGEMM(OPKIT_L3_NO_TRANSPOSE, OPKIT_L3_TRANSPOSE,
        N, M, K,
        alpha, B, N,
        A, M,
        beta, C, N);

    // Equivalently,
    // cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
    //     M, N, K,
    //     alpha, A, M,
    //     B, N,
    //     beta, C, N);
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
    const T alpha = 1.0, const T beta = 0.0)
{
    const size_t A_INC = M * K;
    const size_t B_INC = K * N;
    const size_t C_INC = M * N;

    T* a = (T*) A;
    T* b = (T*) B;
    T* c = (T*) C;

    for (size_t i = 0; i < numChannels; ++i)
    {
        mmMultiply(a, b, c, M, N, K, alpha, beta);
        a += A_INC;
        b += B_INC;
        c += C_INC;
    }
}

// Computes y = alpha * A * x + beta * y, where A is an M x N
// matrix, x is a vector of size N, y is a vector of size M,
// and alpha and beta are scalars.
// xInc and yInc can be adjusted if the vectors are stored
// in an interlaced format.
inline void mvMultiply(const double* A, const double* x, double* y,
    const size_t M, const size_t N,
    const double alpha = 1.0, const double beta = 0.0,
    const int xInc = 1, const int yInc = 1)
{
    USE_ONE_CORE();
    OPKIT_DGEMV(OPKIT_L2_TRANSPOSE,
        N, M,
        alpha, A, N,
        x, xInc,
        beta, y, yInc);

    // Equivalently,
    // cblas_dgemv(CblasRowMajor, CblasNoTrans,
    // M, N,
    // alpha, A, N,
    // x, xInc,
    // beta, y, yInc);
}

// Computes y = alpha * A * x + beta * y, where A is an M x N
// matrix, x is a vector of size N, y is a vector of size M,
// and alpha and beta are scalars.
// xInc and yInc can be adjusted if the vectors are stored
// in an interlaced format.
inline void mvMultiply(const float* A, const float* x, float* y,
    const size_t M, const size_t N,
    const float alpha = 1.0f, const float beta = 0.0f,
    const int xInc = 1, const int yInc = 1)
{
    USE_ONE_CORE();
    OPKIT_SGEMV(OPKIT_L2_TRANSPOSE,
        N, M,
        alpha, A, N,
        x, xInc,
        beta, y, yInc);

    // Equivalently,
    // cblas_sgemv(CblasRowMajor, CblasNoTrans,
    // M, N,
    // alpha, A, N,
    // x, xInc,
    // beta, y, yInc);
}

// Computes y = alpha * A^T * x + beta * y, where A is an M x N
// matrix, x is a vector of size M, y is a vector of size N,
// and alpha and beta are scalars.
// xInc and yInc can be adjusted if the vectors are stored
// in an interlaced format.
inline void mtvMultiply(const double* A, const double* x, double* y,
    const size_t M, const size_t N,
    const double alpha = 1.0, const double beta = 0.0,
    const int xInc = 1, const int yInc = 1)
{
    USE_ONE_CORE();
    OPKIT_DGEMV(OPKIT_L2_NO_TRANSPOSE,
        N, M,
        alpha, A, N,
        x, xInc,
        beta, y, yInc);

    // Equivalently,
    // cblas_dgemv(CblasRowMajor, CblasTrans,
    // M, N,
    // alpha, A, N,
    // x, xInc,
    // beta, y, yInc);
}

// Computes y = alpha * A^T * x + beta * y, where A is an M x N
// matrix, x is a vector of size M, y is a vector of size N,
// and alpha and beta are scalars.
// xInc and yInc can be adjusted if the vectors are stored
// in an interlaced format.
inline void mtvMultiply(const float* A, const float* x, float* y,
    const size_t M, const size_t N,
    const float alpha = 1.0f, const float beta = 0.0f,
    const int xInc = 1, const int yInc = 1)
{
    USE_ONE_CORE();
    OPKIT_SGEMV(OPKIT_L2_NO_TRANSPOSE,
        N, M,
        alpha, A, N,
        x, xInc,
        beta, y, yInc);

    // Equivalently,
    // cblas_sgemv(CblasRowMajor, CblasTrans,
    // M, N,
    // alpha, A, N,
    // x, xInc,
    // beta, y, yInc);
}

// Computes y = alpha * A * x + beta * y, where A is an N x N
// symmetric matrix, x is a vector of size N, y is a vector of size N,
// and alpha and beta are scalars.
// xInc and yInc can be adjusted if the vectors are stored
// in an interlaced format.
inline void symmetricMvMultiply(const double* A, const double* x, double* y,
    const size_t N, const double alpha = 1.0, const double beta = 0.0,
    const int xInc = 1, const int yInc = 1)
{
    USE_ONE_CORE();

    // NOTE: Either row-major order or column-major order can be used for
    // symmetric matrices.
    OPKIT_DSYMV(OPKIT_UPPER,
        N, alpha,
        A, N,
        x, xInc,
        beta, y, yInc);
}

// Computes y = alpha * A * x + beta * y, where A is an N x N
// symmetric matrix, x is a vector of size N, y is a vector of size N,
// and alpha and beta are scalars.
// xInc and yInc can be adjusted if the vectors are stored
// in an interlaced format.
inline void symmetricMvMultiply(const float* A, const float* x, float* y,
    const size_t N, const float alpha = 1.0f, const float beta = 0.0f,
    const int xInc = 1, const int yInc = 1)
{
    USE_ONE_CORE();

    // NOTE: Either row-major order or column-major order can be used for
    // symmetric matrices.
    OPKIT_SSYMV(OPKIT_UPPER,
        N, alpha,
        A, N,
        x, xInc,
        beta, y, yInc);
}

// Adds alpha * x * y^T to A, where x is a vector of size M,
// y is a vector of size N, A is a M x N matrix, and alpha is
// a scalar. When A is initialized to 0's, this calculates the
// vector outer product between x and y. Otherwise, it performs
// a rank-1 update of A.
// xInc and yInc can be adjusted if the vectors are stored
// in an interlaced format.
inline void outerProduct(const double* x, const double* y, double* A,
    const size_t M, const size_t N, const double alpha = 1.0,
    const int xInc = 1, const int yInc = 1)
{
    USE_ONE_CORE();
    OPKIT_DGER(N, M,
        alpha, y, yInc,
        x, xInc,
        A, N);

    // Equivalently,
    // cblas_dger(CblasRowMajor, M, N,
    // alpha, x, xInc,
    // y, yInc,
    // A, N);
}

// Adds alpha * x * y^T to A, where x is a vector of size M,
// y is a vector of size N, A is a M x N matrix, and alpha is
// a scalar. When A is initialized to 0's, this calculates the
// vector outer product between x and y. Otherwise, it performs
// a rank-1 update of A.
// xInc and yInc can be adjusted if the vectors are stored
// in an interlaced format.
inline void outerProduct(const float* x, const float* y, float* A,
    const size_t M, const size_t N, const float alpha = 1.0f,
    const int xInc = 1, const int yInc = 1)
{
    USE_ONE_CORE();
    OPKIT_SGER(N, M,
        alpha, y, yInc,
        x, xInc,
        A, N);

    // Equivalently,
    // cblas_sger(CblasRowMajor, M, N,
    // alpha, x, xInc,
    // y, yInc,
    // A, N);
}

// Computes y += alpha * x, where x is a vector of size N,
// y is a vector of size N, and alpha is a scalar.
// xInc and yInc can be adjusted if the vectors are stored
// in an interlaced format.
inline void vAdd(const double* x, double* y,
    const size_t N, const double alpha = 1.0,
    const int xInc = 1, const int yInc = 1)
{
    USE_ONE_CORE();
    OPKIT_DAXPY(N, alpha, x, xInc, y, yInc);
}

// Computes y += alpha * x, where x is a vector of size N,
// y is a vector of size N, and alpha is a scalar.
// xInc and yInc can be adjusted if the vectors are stored
// in an interlaced format.
inline void vAdd(const float* x, float* y,
    const size_t N, const float alpha = 1.0f,
    const int xInc = 1, const int yInc = 1)
{
    USE_ONE_CORE();
    OPKIT_SAXPY(N, alpha, x, xInc, y, yInc);
}

// Computes x = alpha * x, where x is a vector of size N and
// alpha is a scalar. xInc can be adjusted if the vector is
// stored in an interlaced format.
inline void vScale(double* x, const double alpha,
    const size_t N, const int xInc = 1)
{
    USE_ONE_CORE();
    OPKIT_DSCAL(N, alpha, x, xInc);
}

// Computes x = alpha * x, where x is a vector of size N and
// alpha is a scalar. xInc can be adjusted if the vector is
// stored in an interlaced format.
inline void vScale(float* x, const float alpha,
    const size_t N, const int xInc = 1)
{
    USE_ONE_CORE();
    OPKIT_SSCAL(N, alpha, x, xInc);
}

// Returns the index where the maximum element is found in the vector x of
// size N. xInc can be adjusted if the vector is
// stored in an interlaced format.
inline size_t vMaxIndex(const double* x, const size_t N, const int xInc = 1)
{
    USE_ONE_CORE();
    return OPKIT_IDAMAX(N, x, xInc);
}

// Returns the index where the maximum element is found in the vector x of
// size N. xInc can be adjusted if the vector is
// stored in an interlaced format.
inline size_t vMaxIndex(const float* x, const size_t N, const int xInc = 1)
{
    USE_ONE_CORE();
    return OPKIT_ISAMAX(N, x, xInc);
}

// Copies the contents of x into y, where x and y are vectors of size N.
// xInc and yInc can be adjusted if the vectors are stored in an
// interlaced format.
inline void vCopy(const double* x, double* y, const size_t N,
    const int xInc = 1, const int yInc = 1)
{
    USE_ONE_CORE();
    OPKIT_DCOPY(N, x, xInc, y, yInc);
}

// Copies the contents of x into y, where x and y are vectors of size N.
// xInc and yInc can be adjusted if the vectors are stored in an
// interlaced format.
inline void vCopy(const float* x, float* y, const size_t N,
    const int xInc = 1, const int yInc = 1)
{
    USE_ONE_CORE();
    OPKIT_SCOPY(N, x, xInc, y, yInc);
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
    // Save some useful values
    const size_t NUM_HORIZONTAL_BLOCKS =
        ((destWidth - windowWidth + 2*xPad) / xStride) + 1;
    const size_t NUM_VERTICAL_BLOCKS =
        ((destHeight - windowHeight + 2*yPad) / yStride) + 1;
    const size_t SRC_ROWS  = NUM_HORIZONTAL_BLOCKS * NUM_VERTICAL_BLOCKS;
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


 #endif

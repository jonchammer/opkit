/*
 * File:   Acceleration.h
 * Author: Jon C. Hammer
 *
 * Created on December 11, 2016, 2:00 PM
 */

 #ifndef ACCELERATION_H
 #define ACCELERATION_H

#include <cblas.h>

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
    #include <nvblas.h>

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

    // Computes C = alpha * A^T * B + beta * C, where A is an K x M
    // matrix, B is an K x N matrix, C is an M x N matrix, alpha is
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

    // Computes C = alpha * A^T * B + beta * C, where A is an K x M
    // matrix, B is an K x N matrix, C is an M x N matrix, alpha is
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
        OPKIT_SGER(N, N,
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
};


 #endif

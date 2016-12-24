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
    // For BLAS reference see:
    // https://software.intel.com/sites/default/files/managed/ff/c8/mkl-2017-developer-reference-c_0.pdf

    // Computes C = alpha * A * B + beta * C, where A is an M x K
    // matrix, B is a K x N matrix, C is an M x N matrix, alpha is
    // a scalar, and beta is a scalar.
    inline void mmMultiply(const double* A, const double* B, double* C,
        const size_t M, const size_t N, const size_t K,
        const double alpha = 1.0, const double beta = 0.0)
    {
        // If using OpenBLAS, we only want to use as many threads
        // as possible for expensive computations.
        #ifdef OPENBLAS_CONFIG_H
            openblas_set_num_threads(openblas_get_num_procs());
        #endif

        // Parameters:
        // 1.  Row-major or Col-major
        // 2.  A transpose
        // 3.  B transpose
        // 4.  M
        // 5.  N
        // 6.  K
        // 7.  Alpha
        // 8.  A's data
        // 9.  A's stride
        // 10. B's data
        // 11. B's stride
        // 12. Beta
        // 13. C's data
        // 14. C's stride
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            M, N, K,
            alpha, A, K,
            B, N,
            beta, C, N);
    }

    // Computes C = alpha * A * B + beta * C, where A is an M x K
    // matrix, B is a K x N matrix, C is an M x N matrix, alpha is
    // a scalar, and beta is a scalar.
    inline void mmMultiply(const float* A, const float* B, float* C,
        const size_t M, const size_t N, const size_t K,
        const float alpha = 1.0f, const float beta = 0.0f)
    {
        // If using OpenBLAS, we only want to use as many threads
        // as possible for expensive computations.
        #ifdef OPENBLAS_CONFIG_H
            openblas_set_num_threads(openblas_get_num_procs());
        #endif

        // Parameters:
        // 1.  Row-major or Col-major
        // 2.  A transpose
        // 3.  B transpose
        // 4.  M
        // 5.  N
        // 6.  K
        // 7.  Alpha
        // 8.  A's data
        // 9.  A's stride
        // 10. B's data
        // 11. B's stride
        // 12. Beta
        // 13. C's data
        // 14. C's stride
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            M, N, K,
            alpha, A, K,
            B, N,
            beta, C, N);
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
        // If using OpenBLAS, we only want to use 1 thread for cheap computations.
        #ifdef OPENBLAS_CONFIG_H
            openblas_set_num_threads(1);
        #endif

        // Parameters:
        // 1.  Row-major or Col-major
        // 2.  A transpose
        // 3.  M
        // 4.  N
        // 5.  Alpha
        // 6.  A's data
        // 7.  A's stride
        // 8.  x's data
        // 9.  x's increment (1)
        // 10. Beta
        // 11. y's data
        // 12. y's incrment (1)
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
        M, N,
        alpha, A, N,
        x, xInc,
        beta, y, yInc);
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
        // If using OpenBLAS, we only want to use 1 thread for cheap computations.
        #ifdef OPENBLAS_CONFIG_H
            openblas_set_num_threads(1);
        #endif

        // Parameters:
        // 1.  Row-major or Col-major
        // 2.  A transpose
        // 3.  M
        // 4.  N
        // 5.  Alpha
        // 6.  A's data
        // 7.  A's stride
        // 8.  x's data
        // 9.  x's increment (1)
        // 10. Beta
        // 11. y's data
        // 12. y's incrment (1)
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
        M, N,
        alpha, A, N,
        x, xInc,
        beta, y, yInc);
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
        // If using OpenBLAS, we only want to use 1 thread for cheap computations.
        #ifdef OPENBLAS_CONFIG_H
            openblas_set_num_threads(1);
        #endif

        // Parameters:
        // 1.  Row-major or Col-major
        // 2.  A transpose
        // 3.  M
        // 4.  N
        // 5.  Alpha
        // 6.  A's data
        // 7.  A's stride
        // 8.  x's data
        // 9.  x's increment (1)
        // 10. Beta
        // 11. y's data
        // 12. y's incrment (1)
        cblas_dgemv(CblasRowMajor, CblasTrans,
        M, N,
        alpha, A, N,
        x, xInc,
        beta, y, yInc);
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
        // If using OpenBLAS, we only want to use 1 thread for cheap computations.
        #ifdef OPENBLAS_CONFIG_H
            openblas_set_num_threads(1);
        #endif

        // Parameters:
        // 1.  Row-major or Col-major
        // 2.  A transpose
        // 3.  M
        // 4.  N
        // 5.  Alpha
        // 6.  A's data
        // 7.  A's stride
        // 8.  x's data
        // 9.  x's increment (1)
        // 10. Beta
        // 11. y's data
        // 12. y's incrment (1)
        cblas_sgemv(CblasRowMajor, CblasTrans,
        M, N,
        alpha, A, N,
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
        // If using OpenBLAS, we only want to use 1 thread for cheap computations.
        #ifdef OPENBLAS_CONFIG_H
            openblas_set_num_threads(1);
        #endif

        // Parameters:
        // 1.  Row-major or Col-major
        // 2.  M
        // 3.  N
        // 4.  Alpha
        // 5.  x's data
        // 6.  x's increment (1)
        // 7.  y's data
        // 8.  y's increment (1)
        // 9.  A's data
        // 10. A's leading dimension (N)
        cblas_dger(CblasRowMajor, M, N,
        alpha, x, xInc,
        y, yInc,
        A, N);
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
        // If using OpenBLAS, we only want to use 1 thread for cheap computations.
        #ifdef OPENBLAS_CONFIG_H
            openblas_set_num_threads(1);
        #endif

        // Parameters:
        // 1.  Row-major or Col-major
        // 2.  M
        // 3.  N
        // 4.  Alpha
        // 5.  x's data
        // 6.  x's increment (1)
        // 7.  y's data
        // 8.  y's increment (1)
        // 9.  A's data
        // 10. A's leading dimension (N)
        cblas_sger(CblasRowMajor, M, N,
        alpha, x, xInc,
        y, yInc,
        A, N);
    }

    // Computes y += alpha * x, where x is a vector of size N,
    // y is a vector of size N, and alpha is a scalar.
    // xInc and yInc can be adjusted if the vectors are stored
    // in an interlaced format.
    inline void vAdd(const double* x, double* y,
        const size_t N, const double alpha = 1.0,
        const int xInc = 1, const int yInc = 1)
    {
        // If using OpenBLAS, we only want to use 1 thread for cheap computations.
        #ifdef OPENBLAS_CONFIG_H
            openblas_set_num_threads(1);
        #endif

        // Parameters
        // 1. Size of vectors x and y
        // 2. Scalar 'a'
        // 3. x's data
        // 4. x's increment (1)
        // 5. y's data
        // 6. y's increment (1)
        cblas_daxpy(N, alpha, x, xInc, y, yInc);
    }

    // Computes y += alpha * x, where x is a vector of size N,
    // y is a vector of size N, and alpha is a scalar.
    // xInc and yInc can be adjusted if the vectors are stored
    // in an interlaced format.
    inline void vAdd(const float* x, float* y,
        const size_t N, const float alpha = 1.0f,
        const int xInc = 1, const int yInc = 1)
    {
        // If using OpenBLAS, we only want to use 1 thread for cheap computations.
        #ifdef OPENBLAS_CONFIG_H
            openblas_set_num_threads(1);
        #endif

        // Parameters
        // 1. Size of vectors x and y
        // 2. Scalar 'a'
        // 3. x's data
        // 4. x's increment (1)
        // 5. y's data
        // 6. y's increment (1)
        cblas_saxpy(N, alpha, x, xInc, y, yInc);
    }

    // Computes x = alpha * x, where x is a vector of size N and
    // alpha is a scalar. xInc can be adjusted if the vector is
    // stored in an interlaced format.
    inline void vScale(double* x, const double alpha,
        const size_t N, const int xInc = 1)
    {
        // If using OpenBLAS, we only want to use 1 thread for cheap computations.
        #ifdef OPENBLAS_CONFIG_H
            openblas_set_num_threads(1);
        #endif

        // Parameters
        // 1. Size of vector x
        // 2. Scalar alpha
        // 3. x's data
        // 4. x's increment (1)
        cblas_dscal(N, alpha, x, xInc);
    }

    // Computes x = alpha * x, where x is a vector of size N and
    // alpha is a scalar. xInc can be adjusted if the vector is
    // stored in an interlaced format.
    inline void vScale(float* x, const float alpha,
        const size_t N, const int xInc = 1)
    {
        // If using OpenBLAS, we only want to use 1 thread for cheap computations.
        #ifdef OPENBLAS_CONFIG_H
            openblas_set_num_threads(1);
        #endif

        // Parameters
        // 1. Size of vector x
        // 2. Scalar alpha
        // 3. x's data
        // 4. x's increment (1)
        cblas_sscal(N, alpha, x, xInc);
    }

    // Returns the index where the maximum element is found in the vector x of
    // size N. xInc can be adjusted if the vector is
    // stored in an interlaced format.
    inline size_t vMaxIndex(const double* x, const size_t N, const int xInc = 1)
    {
        // If using OpenBLAS, we only want to use 1 thread for cheap computations.
        #ifdef OPENBLAS_CONFIG_H
            openblas_set_num_threads(1);
        #endif

        // Parameters
        // 1. Size of vector x
        // 2. x's data
        // 3. x's increment (1)
        return cblas_idamax(N, x, xInc);
    }

    // Returns the index where the maximum element is found in the vector x of
    // size N. xInc can be adjusted if the vector is
    // stored in an interlaced format.
    inline size_t vMaxIndex(const float* x, const size_t N, const int xInc = 1)
    {
        // If using OpenBLAS, we only want to use 1 thread for cheap computations.
        #ifdef OPENBLAS_CONFIG_H
            openblas_set_num_threads(1);
        #endif

        // Parameters
        // 1. Size of vector x
        // 2. x's data
        // 3. x's increment (1)
        return cblas_isamax(N, x, xInc);
    }

    // Copies the contents of x into y, where x and y are vectors of size N.
    // xInc and yInc can be adjusted if the vectors are stored in an
    // interlaced format.
    inline void vCopy(const double* x, double* y, const size_t N,
        const int xInc = 1, const int yInc = 1)
    {
        // If using OpenBLAS, we only want to use 1 thread for cheap computations.
        #ifdef OPENBLAS_CONFIG_H
            openblas_set_num_threads(1);
        #endif

        // Parameters
        // 1. Size of vectors x and y
        // 2. x's data
        // 3. x's increment (1)
        // 4. y's data
        // 5. y's increment (1)
        cblas_dcopy(N, x, xInc, y, yInc);
    }

    // Copies the contents of x into y, where x and y are vectors of size N.
    // xInc and yInc can be adjusted if the vectors are stored in an
    // interlaced format.
    inline void vCopy(const float* x, float* y, const size_t N,
        const int xInc = 1, const int yInc = 1)
    {
        // If using OpenBLAS, we only want to use 1 thread for cheap computations.
        #ifdef OPENBLAS_CONFIG_H
            openblas_set_num_threads(1);
        #endif

        // Parameters
        // 1. Size of vectors x and y
        // 2. x's data
        // 3. x's increment (1)
        // 4. y's data
        // 5. y's increment (1)
        cblas_scopy(N, x, xInc, y, yInc);
    }
};


 #endif

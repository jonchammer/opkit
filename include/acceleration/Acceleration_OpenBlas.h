#ifndef ACCELERATION_OPEN_BLAS_H
#define ACCELERATION_OPEN_BLAS_H

#include "Acceleration_CPU.h"
#include <cblas.h>

namespace opkit
{

// For BLAS reference see:
// https://software.intel.com/sites/default/files/managed/ff/c8/mkl-2017-developer-reference-c_0.pdf

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

#define USE_ALL_CORES() openblas_set_num_threads(openblas_get_num_procs())
#define USE_ONE_CORE()  openblas_set_num_threads(1)

template <class T>
struct Acceleration_OpenBlas {};

template <>
struct Acceleration_OpenBlas<double> : public Acceleration_CPU<double>
{
    static void mmMultiply(const double* A, const double* B, double* C,
        const size_t M, const size_t N, const size_t K,
        const size_t ldA, const size_t ldB, const size_t ldC,
        const double alpha, const double beta)
    {
        USE_ONE_CORE();
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
            N, M, K,
            alpha, B, ldB,
            A, ldA,
            beta, C, ldC);
    }

    static void mmtMultiply(const double* A, const double* B, double* C,
        const size_t M, const size_t N, const size_t K,
        const size_t ldA, const size_t ldB, const size_t ldC,
        const double alpha, const double beta)
    {
        USE_ONE_CORE();
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
            N, M, K,
            alpha, B, ldB,
            A, ldA,
            beta, C, ldC);
    }

    static void mtmMultiply(const double* A, const double* B, double* C,
        const size_t M, const size_t N, const size_t K,
        const size_t ldA, const size_t ldB, const size_t ldC,
        const double alpha, const double beta)
    {
        USE_ONE_CORE();
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
            N, M, K,
            alpha, B, ldB,
            A, ldA,
            beta, C, ldC);
    }

    static void mvMultiply(const double* A, const double* x, double* y,
        const size_t M, const size_t N,
        const double alpha, const double beta,
        const int xInc, const int yInc)
    {
        USE_ONE_CORE();
        cblas_dgemv(CblasColMajor, CblasTrans,
            N, M,
            alpha, A, N,
            x, xInc,
            beta, y, yInc);
    }

    static void mtvMultiply(const double* A, const double* x, double* y,
        const size_t M, const size_t N,
        const double alpha, const double beta,
        const int xInc, const int yInc)
    {
        USE_ONE_CORE();
        cblas_dgemv(CblasColMajor, CblasNoTrans,
            N, M,
            alpha, A, N,
            x, xInc,
            beta, y, yInc);
    }

    static void symmetricMvMultiply(const double* A, const double* x, double* y,
        const size_t N, const double alpha, const double beta,
        const int xInc, const int yInc)
    {
        USE_ONE_CORE();

        // NOTE: Either row-major order or column-major order can be used for
        // symmetric matrices.
        cblas_dsymv(CblasColMajor, CblasUpper,
            N, alpha,
            A, N,
            x, xInc,
            beta, y, yInc);
    }

    static void outerProduct(const double* x, const double* y, double* A,
        const size_t M, const size_t N, const double alpha,
        const int xInc, const int yInc)
    {
        USE_ONE_CORE();
        cblas_dger(CblasColMajor, N, M,
            alpha, y, yInc,
            x, xInc,
            A, N);
    }

    static void vAdd(const double* x, double* y,
        const size_t N, const double alpha,
        const int xInc, const int yInc)
    {
        USE_ONE_CORE();
        cblas_daxpy(N, alpha, x, xInc, y, yInc);
    }

    static void vScale(double* x, const double alpha,
        const size_t N, const int xInc)
    {
        USE_ONE_CORE();
        cblas_dscal(N, alpha, x, xInc);
    }

    static size_t vMaxIndex(const double* x, const size_t N, const int xInc)
    {
        USE_ONE_CORE();
        return cblas_idamax(N, x, xInc);
    }

    static void vCopy(const double* x, double* y, const size_t N,
        const int xInc, const int yInc)
    {
        USE_ONE_CORE();
        cblas_dcopy(N, x, xInc, y, yInc);
    }

    static void matrixAdd(const double* A, double* C,
        const size_t M, const size_t N,
        const size_t aStrideX, const size_t aStrideY,
        const size_t cStrideX, const size_t cStrideY,
        const double alpha, const double beta)
    {
        USE_ONE_CORE();

        // NOTE: We don't use geadd directly because it does not work when
        // aStrideY or cStrideY is 0.

        const double* aPtr = A;
              double* cPtr = C;
        for (size_t i = 0; i < M; ++i)
        {
            axpby(aPtr, cPtr, N, alpha, beta, aStrideX, cStrideX);
            aPtr += aStrideY;
            cPtr += cStrideY;
        }
    }

    static void axpby(const double* x, double* y, const size_t N,
        const double alpha, const double beta, const size_t xInc, const size_t yInc)
    {
        cblas_daxpby(N, alpha, x, xInc, beta, y, yInc);
    }
};

template <>
struct Acceleration_OpenBlas<float> : public Acceleration_CPU<float>
{
    static void mmMultiply(const float* A, const float* B, float* C,
        const size_t M, const size_t N, const size_t K,
        const size_t ldA, const size_t ldB, const size_t ldC,
        const float alpha, const float beta)
    {
        USE_ONE_CORE();
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
            N, M, K,
            alpha, B, ldB,
            A, ldA,
            beta, C, ldC);
    }

    static void mmtMultiply(const float* A, const float* B, float* C,
        const size_t M, const size_t N, const size_t K,
        const size_t ldA, const size_t ldB, const size_t ldC,
        const float alpha, const float beta)
    {
        USE_ONE_CORE();
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
            N, M, K,
            alpha, B, ldB,
            A, ldA,
            beta, C, ldC);
    }

    static void mtmMultiply(const float* A, const float* B, float* C,
        const size_t M, const size_t N, const size_t K,
        const size_t ldA, const size_t ldB, const size_t ldC,
        const float alpha, const float beta)
    {
        USE_ONE_CORE();
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
            N, M, K,
            alpha, B, ldB,
            A, ldA,
            beta, C, ldC);
    }

    static void mvMultiply(const float* A, const float* x, float* y,
        const size_t M, const size_t N,
        const float alpha, const float beta,
        const int xInc, const int yInc)
    {
        USE_ONE_CORE();
        cblas_sgemv(CblasColMajor, CblasTrans,
            N, M,
            alpha, A, N,
            x, xInc,
            beta, y, yInc);
    }

    static void mtvMultiply(const float* A, const float* x, float* y,
        const size_t M, const size_t N,
        const float alpha, const float beta,
        const int xInc, const int yInc)
    {
        USE_ONE_CORE();
        cblas_sgemv(CblasColMajor, CblasNoTrans,
            N, M,
            alpha, A, N,
            x, xInc,
            beta, y, yInc);
    }

    static void symmetricMvMultiply(const float* A, const float* x, float* y,
        const size_t N, const float alpha, const float beta,
        const int xInc, const int yInc)
    {
        USE_ONE_CORE();

        // NOTE: Either row-major order or column-major order can be used for
        // symmetric matrices.
        cblas_ssymv(CblasColMajor, CblasUpper,
            N, alpha,
            A, N,
            x, xInc,
            beta, y, yInc);
    }

    static void outerProduct(const float* x, const float* y, float* A,
        const size_t M, const size_t N, const float alpha,
        const int xInc, const int yInc)
    {
        USE_ONE_CORE();
        cblas_sger(CblasColMajor, N, M,
            alpha, y, yInc,
            x, xInc,
            A, N);
    }

    static void vAdd(const float* x, float* y,
        const size_t N, const float alpha,
        const int xInc, const int yInc)
    {
        USE_ONE_CORE();
        cblas_saxpy(N, alpha, x, xInc, y, yInc);
    }

    static void vScale(float* x, const float alpha,
        const size_t N, const int xInc)
    {
        USE_ONE_CORE();
        cblas_sscal(N, alpha, x, xInc);
    }

    static size_t vMaxIndex(const float* x, const size_t N, const int xInc)
    {
        USE_ONE_CORE();
        return cblas_isamax(N, x, xInc);
    }

    static void vCopy(const float* x, float* y, const size_t N,
        const int xInc, const int yInc)
    {
        USE_ONE_CORE();
        cblas_scopy(N, x, xInc, y, yInc);
    }

    static void matrixAdd(const float* A, float* C,
        const size_t M, const size_t N,
        const size_t aStrideX, const size_t aStrideY,
        const size_t cStrideX, const size_t cStrideY,
        const float alpha, const float beta)
    {
        USE_ONE_CORE();

        // NOTE: We don't use geadd directly because it does not work when
        // aStrideY or cStrideY is 0.

        const float* aPtr = A;
              float* cPtr = C;
        for (size_t i = 0; i < M; ++i)
        {
            axpby(aPtr, cPtr, N, alpha, beta, aStrideX, cStrideX);
            aPtr += aStrideY;
            cPtr += cStrideY;
        }
    }

    static void axpby(const float* x, float* y, const size_t N, const float alpha,
        const float beta, const size_t xInc, const size_t yInc)
    {
        cblas_saxpby(N, alpha, x, xInc, beta, y, yInc);
    }
};

}

#endif

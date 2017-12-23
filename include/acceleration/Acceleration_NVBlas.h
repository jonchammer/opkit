#ifndef ACCELERATION_NVBLAS_H
#define ACCELERATION_NVBLAS_H

#include "Acceleration_OpenBlas.h"
#include <nvblas.h>

namespace opkit
{

template <class T>
struct Acceleration_NVBlas {};

template <>
struct Acceleration_NVBlas<double> : public Acceleration_OpenBlas<double>
{
    static void mmMultiply(const double* A, const double* B, double* C,
        const size_t M, const size_t N, const size_t K,
        const size_t ldA, const size_t ldB, const size_t ldC,
        const double alpha, const double beta)
    {
        int m   = (int) M;
        int n   = (int) N;
        int k   = (int) K;
        int lda = (int) ldA;
        int ldb = (int) ldB;
        int ldc = (int) ldC;

        dgemm("N", "N",
            &n, &m, &k,
            &alpha, B, &ldb,
            A, &lda,
            &beta, C, &ldc);
    }

    static void mmtMultiply(const double* A, const double* B, double* C,
        const size_t M, const size_t N, const size_t K,
        const size_t ldA, const size_t ldB, const size_t ldC,
        const double alpha, const double beta)
    {
        int m   = (int) M;
        int n   = (int) N;
        int k   = (int) K;
        int lda = (int) ldA;
        int ldb = (int) ldB;
        int ldc = (int) ldC;

        dgemm("T", "N",
            &n, &m, &k,
            &alpha, B, &ldb,
            A, &lda,
            &beta, C, &ldc);
    }

    static void mtmMultiply(const double* A, const double* B, double* C,
        const size_t M, const size_t N, const size_t K,
        const size_t ldA, const size_t ldB, const size_t ldC,
        const double alpha, const double beta)
    {
        int m   = (int) M;
        int n   = (int) N;
        int k   = (int) K;
        int lda = (int) ldA;
        int ldb = (int) ldB;
        int ldc = (int) ldC;

        dgemm("N", "T",
            &n, &m, &k,
            &alpha, B, &ldb,
            A, &lda,
            &beta, C, &ldc);
    }
};

template <>
struct Acceleration_NVBlas<float> : public Acceleration_OpenBlas<float>
{
    static void mmMultiply(const float* A, const float* B, float* C,
        const size_t M, const size_t N, const size_t K,
        const size_t ldA, const size_t ldB, const size_t ldC,
        const float alpha, const float beta)
    {
        int m   = (int) M;
        int n   = (int) N;
        int k   = (int) K;
        int lda = (int) ldA;
        int ldb = (int) ldB;
        int ldc = (int) ldC;

        sgemm("N", "N",
            &n, &m, &k,
            &alpha, B, &ldb,
            A, &lda,
            &beta, C, &ldc);
    }

    static void mmtMultiply(const float* A, const float* B, float* C,
        const size_t M, const size_t N, const size_t K,
        const size_t ldA, const size_t ldB, const size_t ldC,
        const float alpha, const float beta)
    {
        int m   = (int) M;
        int n   = (int) N;
        int k   = (int) K;
        int lda = (int) ldA;
        int ldb = (int) ldB;
        int ldc = (int) ldC;

        sgemm("T", "N",
            &n, &m, &k,
            &alpha, B, &ldb,
            A, &lda,
            &beta, C, &ldc);
    }

    static void mtmMultiply(const float* A, const float* B, float* C,
        const size_t M, const size_t N, const size_t K,
        const size_t ldA, const size_t ldB, const size_t ldC,
        const float alpha, const float beta)
    {
        int m   = (int) M;
        int n   = (int) N;
        int k   = (int) K;
        int lda = (int) ldA;
        int ldb = (int) ldB;
        int ldc = (int) ldC;

        sgemm("N", "T",
            &n, &m, &k,
            &alpha, B, &ldb,
            A, &lda,
            &beta, C, &ldc);
    }
};

}

#endif

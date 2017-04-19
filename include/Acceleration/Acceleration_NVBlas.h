#ifndef ACCELERATION_NVBLAS_H
#define ACCELERATION_NVBLAS_H

#include "Acceleration_OpenBlas.h"
#include <nvblas.h>

namespace opkit
{

template <>
struct Acceleration_NVBlas<double> : public Acceleration_OpenBlas<double>
{
    static void mmMultiply(const double* A, const double* B, double* C,
        const size_t M, const size_t N, const size_t K,
        const double alpha, const double beta)
    {
        // TODO: Implement
    }

    static void mmtMultiply(const double* A, const double* B, double* C,
        const size_t M, const size_t N, const size_t K,
        const double alpha, const double beta)
    {
        // TODO: Implement
    }

    static void mtmMultiply(const double* A, const double* B, double* C,
        const size_t M, const size_t N, const size_t K,
        const double alpha, const double beta)
    {
        // TODO: Implement
    }
};

template <>
struct Acceleration_NVBlas<float> : public Acceleration_OpenBlas<float>
{
    static void mmMultiply(const float* A, const float* B, float* C,
        const size_t M, const size_t N, const size_t K,
        const float alpha, const float beta)
    {
        // TODO: Implement
    }

    static void mmtMultiply(const float* A, const float* B, float* C,
        const size_t M, const size_t N, const size_t K,
        const float alpha, const float beta)
    {
        // TODO: Implement
    }

    static void mtmMultiply(const float* A, const float* B, float* C,
        const size_t M, const size_t N, const size_t K,
        const float alpha, const float beta)
    {
        // TODO: Implement
    }
};

}

#endif

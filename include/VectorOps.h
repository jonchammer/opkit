#ifndef VECTOR_OPS_H
#define VECTOR_OPS_H

#include "Acceleration.h"

template <class T>
T mean(const T* vec, const size_t length)
{
    T sum{};
    for (size_t i = 0; i < length; ++i)
    {
        sum += vec[i];
    }
    return sum / length;
}

template <class T>
T variance(const T* vec, const size_t length)
{
    T avg = mean(vec, length);
    T sum{};
    for (size_t i = 0; i < length; ++i)
    {
        T temp = vec[i] - avg;
        sum   += temp * temp;
    }

    return sum / length;
}

template <class T>
T magnitude(const T* vec, const size_t length)
{
    T mag = T{};
    for (size_t i = 0; i < length; ++i)
    {
        mag += vec[i] * vec[i];
    }
    return std::sqrt(mag);
}

template <class T>
void normalize(T* vec, const size_t length)
{
    T invMag = T{1.0} / magnitude(vec, length);
    for (size_t i = 0; i < length; ++i)
        vec[i] *= invMag;
}

template <class T>
T min(const T* vec, const size_t length)
{
    T m = vec[0];
    for (size_t i = 1; i < length; ++i)
    {
        if (vec[i] < m)
            m = vec[i];
    }
    return m;
}

template <class T>
T max(const T* vec, const size_t length)
{
    T m = vec[0];
    for (size_t i = 1; i < length; ++i)
    {
        if (vec[i] > m)
            m = vec[i];
    }
    return m;
}

template <class T>
T effectiveDensity(const T* vec, const size_t length, const T limit = 1E-3)
{
    size_t count = 0;
    for (size_t i = 0; i < length; ++i)
    {
        if (std::abs(vec[i]) > limit)
            ++count;
    }

    return T(count) / T(length);
}

#endif

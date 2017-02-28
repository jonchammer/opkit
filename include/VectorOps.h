#ifndef VECTOR_OPS_H
#define VECTOR_OPS_H

#include <vector>
using std::vector;

template <class T>
T mean(const vector<T>& vec)
{
    T sum{};
    for (size_t i = 0; i < vec.size(); ++i)
    {
        sum += vec[i];
    }
    return sum / vec.size();
}

template <class T>
T variance(const vector<T>& vec)
{
    T avg = mean(vec);
    T sum{};
    for (size_t i = 0; i < vec.size(); ++i)
    {
        T temp = vec[i] - avg;
        sum   += temp * temp;
    }

    return sum / vec.size();
}

template <class T>
T magnitude(const vector<T>& vec)
{
    T mag = T{};
    for (size_t i = 0; i < vec.size(); ++i)
    {
        mag += vec[i] * vec[i];
    }
    return std::sqrt(mag);
}

template <class T>
void normalize(const vector<T>& vec)
{
    T mag = magnitude(vec);
    for (size_t i = 0; i < vec.size(); ++i)
    {
        vec[i] /= mag;
    }
}

#endif

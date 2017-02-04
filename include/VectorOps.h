#ifndef VECTOR_OPS_H
#define VECTOR_OPS_H

#include <vector>
using std::vector;

template <class T>
T magnitude(vector<T>& vec)
{
    T mag = T{};
    for (size_t i = 0; i < vec.size(); ++i)
    {
        mag += vec[i] * vec[i];
    }
    return std::sqrt(mag);
}

template <class T>
void normalize(vector<T>& vec)
{
    T mag = magnitude(vec);
    for (size_t i = 0; i < vec.size(); ++i)
    {
        vec[i] /= mag;
    }
}

#endif

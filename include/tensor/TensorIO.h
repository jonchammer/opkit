#ifndef TENSOR_IO_H
#define TENSOR_IO_H

#include "tensor/Tensor.h"
#include "tensor/TensorMath.h"

namespace tensorlib
{

// Print the tensor when both the width and the precision are known
template <class T>
std::string to_string(const Tensor<T>& A, const size_t width, const size_t precision)
{
    std::stringstream ss;
    switch (A.rank())
    {
        // 1 row
        case 0: case 1:
            ss << "[";
            for (auto& elem : A)
                ss << std::setw(width) << std::fixed
                   << std::setprecision(precision) << elem;
            ss << " ]";
            break;

        // Multiple rows
        case 2:
            for (size_t i = 0; i < A.shape(0) - 1; ++i)
                ss << to_string(A({i}), width, precision) << std::endl;
            ss << to_string(A({A.shape(0) - 1}), width, precision);
            break;

        // Multiple tables
        default:

            SmallVector truncatedShape(A.shape());
            truncatedShape.pop_back();
            truncatedShape.pop_back();
            SmallVector indices(truncatedShape.size());

            while (true)
            {
                ss << "( ";
                for (auto& elem : indices)
                    ss << elem << ", ";
                ss << "*, * ) = " << std::endl;
                ss << to_string(A(indices.begin(), indices.end()), width, precision) << std::endl;

                // Update the index (N-ary counter)
                int dim = indices.size() - 1;
                indices[dim]++;
                while (dim > 0 && indices[dim] >= truncatedShape[dim])
                {
                    indices[dim] = 0;
                    indices[dim - 1]++;
                    --dim;
                }

                if (indices[0] < truncatedShape[0])
                    ss << std::endl;
                else
                {
                    ss << "Shape: " << to_string(A.shape()) << std::endl;
                    break;
                }
            }

            break;

    }
    return ss.str();
}

// When A is a floating point type, make room for the precision and the width
template <class T>
std::string to_string(const Tensor<T>& A, const size_t precision = 2,
    typename std::enable_if<std::is_floating_point<T>::value >::type* = 0)
{
    // Determine how much space should be used for each element
    T largest  = T(reduceMax(A));
    T smallest = T(reduceMin(A));
    int w1     = std::ceil(std::log10(std::abs(largest)));
    int w2     = std::ceil(std::log10(std::abs(smallest)));
    bool neg   = largest < T{} || smallest < T{};
    int width  = std::max( w1, std::max(w2, 1 + (int) neg) );
    width     += 2 + precision;
    return to_string(A, width, precision);
}

// When A is an integral type, we just need to calculate the optimal width
template <class T>
std::string to_string(const Tensor<T>& A,
    typename std::enable_if<std::is_integral<T>::value >::type* = 0)
{
    // Determine how much space should be used for each element
    T largest    = T(reduceMax(A));
    size_t width = std::log10(largest) + 2;
    return to_string(A, width, 0);
}

// When A is some other type, there's not really anything intelligent we can
// do, so just print it.
template <class T>
std::string to_string(const Tensor<T>& A,
    typename std::enable_if<!std::is_arithmetic<T>::value >::type* = 0)
{
    return to_string(A, 0, 0);
}

// Always print to a stream using optimal width (if possible). If a custom
// width or precision is desired, you should do something like this instead:
//   out << to_string(A, ...);
template <class T>
std::ostream& operator<<(std::ostream& out, const Tensor<T>& A)
{
    out << to_string(A);
    return out;
}

}
#endif

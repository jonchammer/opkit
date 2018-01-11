#ifndef MATH_H
#define MATH_H

#include <string>
#include <random>
#include <chrono>
#include "acceleration/Acceleration.h"
#include "util/Assert.h"
#include "util/Rand.h"
#include "tensor/Tensor.h"
#include "tensor/TensorOps.h"

namespace detail
{

using opkit::SmallVector;
using opkit::Tensor;

// Calculates the broadcast shape between two tensors, similar to NumPy. See:
// https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html
//
// Ex:
// A      (4d array):  8 x 1 x 6 x 1
// B      (3d array):      7 x 1 x 5
// Result (4d array):  8 x 7 x 6 x 5
template <class T>
SmallVector getBroadcastShape(const Tensor<T>& A, const Tensor<T>& B)
{
    SmallVector res(std::max(A.rank(), B.rank()));

    const SmallVector& aShape = A.shape();
    const SmallVector& bShape = B.shape();

    size_t index = res.size() - 1;
    auto aIt     = aShape.rbegin();
    auto bIt     = bShape.rbegin();

    for (; aIt != aShape.rend() && bIt != bShape.rend(); ++aIt, ++bIt, --index)
    {
        if (*aIt == 1 || *bIt == 1 || *aIt == *bIt)
            res[index] = std::max(*aIt, *bIt);
        else ASSERT(false, "Unable to coerce a valid broadcast shape. A = " +
            opkit::to_string(aShape) + ", B = " + opkit::to_string(bShape));
    }

    // Copy the remaining shape elements from A
    while (aIt != aShape.rend())
    {
        res[index] = *aIt;
        --index;
        ++aIt;
    }

    // Copy the remaining shape elements from B
    while (bIt != bShape.rend())
    {
        res[index] = *bIt;
        --index;
        ++bIt;
    }

    return res;
}

// Helper function - Calculates c = f(a, b), where f can be any function
// and a and b are vectors of the same size.
//
// 'f' should have a signature resembling:
//     T f(const T a, const T b)
template <class T, class Func>
void binaryVectorOp(
    Func&& f,
    const T* a, const T* b, T* c,
    const size_t N,
    const size_t aStride, const size_t bStride)
{
    for (size_t x = 0; x < N; ++x)
    {
        *c++ = f(*a, *b);
        a   += aStride;
        b   += bStride;
    }
}

// Helper function - Calculates a op= b, where op can be any function and a and
// b are vectors of the same size.
//
// 'op' should have a signature resembling:
//     void op(T& res, const T newValue)
template <class T, class Func>
void vectorUpdateOp(
    Func&& f,
    T* a, const T* b,
    const size_t N,
    const size_t aStride, const size_t bStride)
{
    for (size_t x = 0; x < N; ++x)
    {
        f(*a, *b);
        a += aStride;
        b += bStride;
    }
}

// Helper function - Calculates c = f(a, b), where f can be any function
// and a and b are matrices of the same size.
//
// 'f' should have a signature resembling:
//     T f(const T a, const T b)
template <class T, class Func>
void binaryMatrixOp(
    Func&& f,
    const T* a, const T* b, T* c,
    const size_t M, const size_t N,
    const size_t aStrideX, const size_t aStrideY,
    const size_t bStrideX, const size_t bStrideY)
{
    for (size_t y = 0; y < M; ++y)
    {
        const T* aRow = a;
        const T* bRow = b;
        for (size_t x = 0; x < N; ++x)
        {
            *c++  = f(*aRow, *bRow);
            aRow += aStrideX;
            bRow += bStrideX;
        }

        a += aStrideY;
        b += bStrideY;
    }
}

// Helper function - Calculates a op= b, where op can be any function and a and
// b are matrices of the same size.
//
// 'op' should have a signature resembling:
//     void op(T& res, const T newValue)
template <class T, class Func>
void matrixUpdateOp(
    Func&& f,
    T* a, const T* b,
    const size_t M, const size_t N,
    const size_t aStrideX, const size_t aStrideY,
    const size_t bStrideX, const size_t bStrideY)
{
    for (size_t y = 0; y < M; ++y)
    {
              T* aRow = a;
        const T* bRow = b;
        for (size_t x = 0; x < N; ++x)
        {
            f(*aRow, *bRow);
            aRow += aStrideX;
            bRow += bStrideX;
        }

        a += aStrideY;
        b += bStrideY;
    }
}

}

namespace opkit
{

// ------------------------------- Utilities -------------------------------- //

// Apply 'f' to every individual element of the input
template <class T, class Func>
Tensor<T> elementwiseFunc(const Tensor<T>& arg, Func&& f)
{
    Tensor<T> res = arg.clone();
    res.apply(std::forward<Func>(f));
    return res;
}

template <class T, class Func>
void elementwiseFunc(Tensor<T>& res, const Tensor<T>& arg, Func&& f)
{
    arg.copy(res);
    res.apply(std::forward<Func>(f));
}

// Faster approximation of e^x when precision isn't as important
template <class T>
T fastExp(T x)
{
    x = 1.0 + x / 1024.0;
    x *= x; x *= x; x *= x; x *= x;
    x *= x; x *= x; x *= x; x *= x;
    x *= x; x *= x;
    return x;
}

// ------------------------ Broadcasting Binary Ops ------------------------- //

// Elementwise binary function that supports broadcasting so the sizes don't
// have to line up exactly.
template <class T, class Func>
Tensor<T> broadcastingBinaryOp(const Tensor<T>& A, const Tensor<T>& B, Func f)
{
    SmallVector& shape = A.shape();
    Tensor<T> res(shape.begin(), shape.end());
    broadcastingBinaryOp(res, A, B, std::forward<Func>(f));
    return res;
}

template <class T, class Func>
void broadcastingBinaryOp(Tensor<T>& res, const Tensor<T>& A, const Tensor<T>& B, Func f)
{
    // Try to use broadcasting
    if (A.shape() != B.shape())
    {
        SmallVector commonShape = detail::getBroadcastShape(A, B);
        return broadcastingBinaryOp(res, expand(A, commonShape.begin(), commonShape.end()),
            expand(B, commonShape.begin(), commonShape.end()), f);
    }

    const SmallVector& shape = A.shape();
    res.resize(shape.begin(), shape.end());

    // Optimization for matrices
    if (A.rank() == 2 /*&& B.rank() == 2*/)
    {
        detail::binaryMatrixOp(std::forward<Func>(f),
            A.data(), B.data(), res.data(),
            A.shape(0), A.shape(1),
            A.stride(1), A.stride(0),
            B.stride(1), B.stride(0));
    }

    // Slower path that will work in general
    else
    {
        // Apply binary op. We take advantage of the fact that 'res' is guaranteed
        // to be continuous to improve performance a bit.
        T* resData = res.data();
        auto itB   = B.begin();

        for (const T& elem : A)
        {
            *resData = f(elem, *itB);
            ++itB; ++resData;
        }
    }
}

// Adds two tensors together. If the two tensors have the same shape, the
// addition is performed elementwise, as expected. If the two tensors DO NOT
// have the same shape, broadcasting will be attempted to see if the two tensors
// can possibly be coerced into having a common shape.
//
// For example,
// [ 1 2 3 ]
// [ 4 5 6 ] + [ x y z ]
// [ 7 8 9 ]
//
// will be added as:
//
// [ 1 2 3 ] + [ x y z ]
// [ 4 5 6 ] + [ x y z ]
// [ 7 8 9 ] + [ x y z ]
template <class T>
Tensor<T> add(const Tensor<T>& A, const Tensor<T>& B)
{
    Tensor<T> res;
    add(res, A, B);
    return res;
}

template <class T>
void add(Tensor<T>& res, const Tensor<T>& A, const Tensor<T>& B)
{
    // Try to use broadcasting
    if (A.shape() != B.shape())
    {
        SmallVector commonShape = detail::getBroadcastShape(A, B);
        return add(res, expand(A, commonShape.begin(), commonShape.end()),
            expand(B, commonShape.begin(), commonShape.end()));
    }

    // Apply binary op. We take advantage of the fact that 'res' is guaranteed
    // to be continuous to improve performance a bit.
    A.copy(res);
    T* resData = res.data();

    // Add B
    if (B.contiguous())
        vAdd(B.data(), resData, A.size());
    else if (B.rank() == 2)
    {
        matrixAdd(B.data(), resData, res.shape(0), res.shape(1),
            B.stride(1), B.stride(0), res.stride(1), res.stride(0), T{1}, T{1});
    }
    else
    {
        auto itB = B.begin();
        for (size_t i = 0; i < res.size(); ++i)
        {
            *resData++ += *itB;
            ++itB;
        }
    }
}

// Subtracts B from A. If the two tensors have the same shape, the
// subtraction is performed elementwise, as expected. If the two tensors DO NOT
// have the same shape, broadcasting will be attempted to see if the two tensors
// can possibly be coerced into having a common shape, as with add().
template <class T>
Tensor<T> sub(const Tensor<T>& A, const Tensor<T>& B)
{
    Tensor<T> res;
    sub(res, A, B);
    return res;
}

template <class T>
void sub(Tensor<T>& res, const Tensor<T>& A, const Tensor<T>& B)
{
    // Try to use broadcasting
    if (A.shape() != B.shape())
    {
        SmallVector commonShape = detail::getBroadcastShape(A, B);
        return sub(res, expand(A, commonShape.begin(), commonShape.end()),
            expand(B, commonShape.begin(), commonShape.end()));
    }

    // Apply binary op. We take advantage of the fact that 'res' is guaranteed
    // to be continuous to improve performance a bit.
    A.copy(res);
    T* resData = res.data();

    // Subtract B
    if (B.contiguous())
        vAdd(B.data(), resData, A.size(), T{-1});
    else if (B.rank() == 2)
    {
        matrixAdd(B.data(), resData, res.shape(0), res.shape(1),
            B.stride(1), B.stride(0), res.stride(1), res.stride(0), T{-1}, T{1});
    }
    else
    {
        auto itB = B.begin();
        for (size_t i = 0; i < res.size(); ++i)
        {
            *resData++ -= *itB;
            ++itB;
        }
    }
}

// Multiplies A and B. If the two tensors have the same shape, the
// multiplication is performed elementwise, as expected. If the two tensors DO NOT
// have the same shape, broadcasting will be attempted to see if the two tensors
// can possibly be coerced into having a common shape, as with add().
//
// NOTE: The broadcasting behavior means that this function is capable of
// calculating scalar products natively.
//
// NOTE: Outer products between two vectors can be calculated in concert with
// the expand() function. Ex: A, shape = [M], B, shape = [N]
//   Tensor<T> result = multiply( A.expand( { M, 1 } ), B );
//
// NOTE: In theory, multiply() can be used to implement an inner product as
// well, using something like this:
//   Tensor<T> innerProduct = reduceSum( multiply( A, B ) );
// Since we have a dedicated innerProduct() function, however, there is little
// need to use a more complicated procedure.
template <class T>
Tensor<T> multiply(const Tensor<T>& A, const Tensor<T>& B)
{
    Tensor<T> res;
    multiply(res, A, B);
    return res;
}

template <class T>
void multiply(Tensor<T>& res, const Tensor<T>& A, const Tensor<T>& B)
{
    // Handle easy cases first
    if (A.size() == 1)
    {
        const T val   = T(A);
        B.copy(res);
        scale(res, val);
    }

    else if (B.size() == 1)
    {
        const T val   = T(B);
        A.copy(res);
        scale(res, val);
    }

    else broadcastingBinaryOp(res, A, B, [](const T& a, const T& b)
    {
        return a * b;
    });
}

// Divides A and B. If the two tensors have the same shape, the
// division is performed elementwise, as expected. If the two tensors DO NOT
// have the same shape, broadcasting will be attempted to see if the two tensors
// can possibly be coerced into having a common shape, as with add().
//
// NOTE: The native / operator is used for calculations, so if T is an integral
// type, integer division will be used. If T is a floating point type, floating
// point division will be used.
template <class T>
Tensor<T> divide(const Tensor<T>& A, const Tensor<T>& B)
{
    Tensor<T> res;
    divide(res, A, B);
    return res;
}

template <class T>
void divide(Tensor<T>& res, const Tensor<T>& A, const Tensor<T>& B)
{
    // Handle easy cases first
    if (A.size() == 1)
    {
        const T val   = T(A);
        B.copy(res);
        res.apply([&val](const T& x)
        {
            return val / x;
        });
    }

    else if (B.size() == 1)
    {
        const T val   = T{1} / T(B);
        A.copy(res);
        scale(res, val);
    }

    else broadcastingBinaryOp(res, A, B, [](const T& a, const T& b)
    {
        return a / b;
    });
}

// Calculates the maximum of A_i and B_i for each individual element in A and B.
template <class T>
Tensor<T> max(const Tensor<T>& A, const Tensor<T>& B)
{
    Tensor<T> res;
    max(res, A, B);
    return res;
}

template <class T>
void max(Tensor<T>& res, const Tensor<T>& A, const Tensor<T>& B)
{
    // Handle easy cases first
    if (A.size() == 1)
    {
        const T val{A};
        B.copy(res);
        res.apply([&val](const T& x)
        {
            return std::max(x, val);
        });
    }

    else if (B.size() == 1)
    {
        const T val{B};
        A.copy(res);
        res.apply([&val](const T& x)
        {
            return std::max(x, val);
        });
    }

    else broadcastingBinaryOp(res, A, B, [](const T& a, const T& b)
    {
        return std::max(a, b);
    });
}

// Calculates the minimum of A_i and B_i for each individual element in A and B.
template <class T>
Tensor<T> min(const Tensor<T>& A, const Tensor<T>& B)
{
    Tensor<T> res;
    min(res, A, B);
    return res;
}

template <class T>
void min(Tensor<T>& res, const Tensor<T>& A, const Tensor<T>& B)
{
    // Handle easy cases first
    if (A.size() == 1)
    {
        const T val{A};
        B.copy(res);
        res.apply([&val](const T& x)
        {
            return std::min(x, val);
        });
    }

    else if (B.size() == 1)
    {
        const T val{B};
        A.copy(res);
        res.apply([&val](const T& x)
        {
            return std::min(x, val);
        });
    }

    else broadcastingBinaryOp(res, A, B, [](const T& a, const T& b)
    {
        return std::min(a, b);
    });
}

// ------------------------------- Boolean Ops ------------------------------ //

template <class T>
Tensor<T> equal(const Tensor<T>& A, const Tensor<T>& B)
{
    return broadcastingBinaryOp(A, B, [](const T& a, const T& b)
    {
        return T(a == b);
    });
}

template <class T>
void equal(Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& B)
{
    broadcastingBinaryOp(y, A, B, [](const T& a, const T& b)
    {
        return T(a == b);
    });
}

template <class T>
Tensor<T> notEqual(const Tensor<T>& A, const Tensor<T>& B)
{
    return broadcastingBinaryOp(A, B, [](const T& a, const T& b)
    {
        return T(a != b);
    });
}

template <class T>
void notEqual(Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& B)
{
    broadcastingBinaryOp(y, A, B, [](const T& a, const T& b)
    {
        return T(a != b);
    });
}

template <class T>
Tensor<T> greater(const Tensor<T>& A, const Tensor<T>& B)
{
    return broadcastingBinaryOp(A, B, [](const T& a, const T& b)
    {
        return T(a > b);
    });
}

template <class T>
void greater(Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& B)
{
    broadcastingBinaryOp(y, A, B, [](const T& a, const T& b)
    {
        return T(a > b);
    });
}

template <class T>
Tensor<T> greaterEqual(const Tensor<T>& A, const Tensor<T>& B)
{
    return broadcastingBinaryOp(A, B, [](const T& a, const T& b)
    {
        return T(a >= b);
    });
}

template <class T>
void greaterEqual(Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& B)
{
    broadcastingBinaryOp(y, A, B, [](const T& a, const T& b)
    {
        return T(a >= b);
    });
}

template <class T>
Tensor<T> less(const Tensor<T>& A, const Tensor<T>& B)
{
    return broadcastingBinaryOp(A, B, [](const T& a, const T& b)
    {
        return T(a < b);
    });
}

template <class T>
void less(Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& B)
{
    broadcastingBinaryOp(y, A, B, [](const T& a, const T& b)
    {
        return T(a < b);
    });
}

template <class T>
Tensor<T> lessEqual(const Tensor<T>& A, const Tensor<T>& B)
{
    return broadcastingBinaryOp(A, B, [](const T& a, const T& b)
    {
        return T(a <= b);
    });
}

template <class T>
void lessEqual(Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& B)
{
    broadcastingBinaryOp(y, A, B, [](const T& a, const T& b)
    {
        return T(a <= b);
    });
}

// ------------------------------- Update Ops ------------------------------- //

// Calculates A op= B, where A, B are tensors with a broadcast shape in common.
//
// The signature of 'op' should be resemble the following:
//     void op(T& a, const T b)
template <class T, class Func>
void updateOp(Tensor<T>& A, const Tensor<T>& B, Func&& f)
{
    // Try to use broadcasting
    if (A.shape() != B.shape())
    {
        SmallVector commonShape = detail::getBroadcastShape(A, B);
        ASSERT(commonShape == A.shape(), "updateOp: Only B's shape can change.");
        updateOp(A, expand(B, commonShape.begin(), commonShape.end()),
            std::forward<Func>(f));
        return;
    }

    // Optimization for vectors
    if (A.rank() == 1 /*&& B.rank() == 1*/)
    {
        detail::vectorUpdateOp(std::forward<Func>(f), A.data(), B.data(),
            A.shape(0), A.stride(0), B.stride(0));
    }

    // Optimization for matrices
    else if (A.rank() == 2 /*&& B.rank() == 2*/)
    {
        detail::matrixUpdateOp(std::forward<Func>(f),
            A.data(), B.data(),
            A.shape(0), A.shape(1),
            A.stride(1), A.stride(0),
            B.stride(1), B.stride(0));
    }

    // General implementation that will always work
    else
    {
        auto bIt = B.begin();
        for (T& elem : A)
        {
            f(elem, *bIt);
            ++bIt;
        }
    }
}

// A += B;
template <class T>
void addTo(Tensor<T>& A, const Tensor<T>& B)
{
    updateOp(A, B, [](T& y, const T& x)
    {
        y += x;
    });
}

// A -= B;
template <class T>
void subFrom(Tensor<T>& A, const Tensor<T>& B)
{
    updateOp(A, B, [](T& y, const T& x)
    {
        y -= x;
    });
}

// A *= B;
template <class T>
void multBy(Tensor<T>& A, const Tensor<T>& B)
{
    updateOp(A, B, [](T& y, const T& x)
    {
        y *= x;
    });
}

// A /= B;
template <class T>
void divBy(Tensor<T>& A, const Tensor<T>& B)
{
    updateOp(A, B, [](T& y, const T& x)
    {
        y /= x;
    });
}

// y += alpha * x
template <class T>
void axpy(Tensor<T>& y, const Tensor<T>& x, const Tensor<T>& alpha)
{
    ASSERT(x.shape() == y.shape(), "X must be the same shape as Y");
    ASSERT(alpha.size() == 1, "Alpha must be a scalar.");

    if (x.contiguous() && y.contiguous())
        vAdd(x.data(), y.data(), y.size(), T(alpha));
    else
    {
        auto xIt = x.begin();
        for (T& elem : y)
        {
            elem += T(alpha) * *xIt;
            ++xIt;
        }
    }
}

// x *= alpha
template <class T, class U>
void scale(Tensor<T>& x, const U alpha)
{
    // Optimization for contiguous tensors
    if (x.contiguous())
        vScale(x.data(), T(alpha), x.size());

    // Generic implementation
    else x.apply([&alpha](const T& x)
    {
        return x * alpha;
    });
}

// ------------------------------ [ Category ]  ----------------------------- //

// Clips each element of the given tensor to the range [min, max]
template <class T, class U>
Tensor<T> clip(const Tensor<T>& A, const U min, const U max)
{
    Tensor<T> res;
    clip(res, A, min, max);
    return res;
}

template <class T, class U>
void clip(Tensor<T>& y, const Tensor<T>& A, const U min, const U max)
{
    A.copy(y);
    y.apply([&min, &max](const T& x)
    {
        return std::max(T{min}, std::min(x, T{max}));
    });
}

// Raises each element of A to the given power.
template <class T, class U>
Tensor<T> pow(const Tensor<T>& A, const U& power)
{
    Tensor<T> res;
    pow(res, A, power);
    return res;
}

template <class T>
Tensor<T> pow(const Tensor<T>& A, const Tensor<T>& power)
{
    ASSERT(power.size() == 1, "power must be a scalar");
    return pow(A, T(power));
}

template <class T, class U>
void pow(Tensor<T>& res, const Tensor<T>& A, const U& power)
{
    A.copy(res);
    res.apply([&power](const T& x)
    {
        return pow(x, T(power));
    });
}

template <class T>
void pow(Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& power)
{
    ASSERT(power.size() == 1, "power must be a scalar");
    pow(y, A, T(power));
}

// Returns the inner product between the two tensors A and B. Both
// tensors must have the same shape. For vectors, this operation
// is usually called a dot product, and for matrices, it's a
// Frobenius inner product. In all cases, the result is a single
// scalar.
template <class T>
Tensor<T> innerProduct(const Tensor<T>& A, const Tensor<T>& B)
{
    Tensor<T> res;
    innerProduct(res, A, B);
    return res;
}

template <class T>
void innerProduct(Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& B)
{
    ASSERT(A.shape() == B.shape(), "Shapes must match");

    T sum{};
    auto itB = B.begin();
    for (const T& elem : A)
    {
        sum += elem * *itB;
        ++itB;
    }

    y.resize({1});
    y = sum;
}
// -------------------------- Vector / Matrix Ops --------------------------- //

// A:      Rank 1, size N
// B:      Rank 2, size N x M
// Result: Rank 1, size N
template <class T>
Tensor<T> vectorMatrixMultiply(const Tensor<T>& A, const Tensor<T>& B)
{
    // TODO Implement
    ASSERT(false, "Not implemented");
}

template <class T>
void vectorMatrixMultiply(Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& B)
{
    // TODO Implement
    ASSERT(false, "Not implemented");
}

// A:      Rank 2, size M x N,
// B:      Rank 1, size N
// Result: Rank 1, size M
template <class T>
Tensor<T> matrixVectorMultiply(const Tensor<T>& A, const Tensor<T>& B)
{
    // TODO Implement
    ASSERT(false, "Not implemented");
}

template <class T>
void matrixVectorMultiply(Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& B)
{
    // TODO Implement
    ASSERT(false, "Not implemented");
}

// Multiplies two tensors together. Both have a rank of exactly 2.
//
// TODO: Support multiplications like: [a, b, m, k] x [a, b, k, n] => [a, b, m, n]
// for tensors with larger ranks.
template <class T>
Tensor<T> matrixMultiply(const Tensor<T>& A, const Tensor<T>& B)
{
    Tensor<T> res;
    matrixMultiply(res, A, B);
    return res;
}

template <class T>
void matrixMultiply(Tensor<T>& res, const Tensor<T>& A, const Tensor<T>& B)
{
    // Check size constraints
    ASSERT(A.rank() == 2 && B.rank() == 2, "A, B must be matrices. shape(A) = " +
        to_string(A.shape()) + ", shape(B) = " + to_string(B.shape()));
    ASSERT(A.shape(1) == B.shape(0), "A.cols() must == B.rows(). " +
        to_string(A.shape()) + " vs. " + to_string(B.shape()));

    const size_t M = A.shape(0);
    const size_t N = B.shape(1);
    const size_t K = A.shape(1);

    res.resize({M, N});
    res.fill(T{0});
    mmMultiply(A.data(), B.data(), res.data(), M, N, K, A.stride(1), A.stride(0),
        B.stride(1), B.stride(0), res.stride(1), res.stride(0));
}

// Multiplies two tensors together. Both have a rank of exactly 2.
// The first matrix is assumed to be transposed.
//
// TODO: Support multiplications like: [a, b, m, k] x [a, b, k, n] => [a, b, m, n]
// for tensors with larger ranks.
template <class T>
Tensor<T> matrixMultiplyT1(const Tensor<T>& A, const Tensor<T>& B)
{
    Tensor<T> res;
    matrixMultiplyT1(res, A, B);
    return res;
}

template <class T>
void matrixMultiplyT1(Tensor<T>& res, const Tensor<T>& A, const Tensor<T>& B)
{
    // Check size constraints
    ASSERT(A.rank() == 2 && B.rank() == 2, "A, B must be matrices");
    ASSERT(A.shape(0) == B.shape(0), "A.cols() must == B.rows(). " +
        to_string(A.shape()) + " vs. " + to_string(B.shape()));

    const size_t M = A.shape(1);
    const size_t N = B.shape(1);
    const size_t K = A.shape(0);

    res.resize({M, N});
    res.fill(T{0});
    mtmMultiply(A.data(), B.data(), res.data(), M, N, K, A.stride(1), A.stride(0),
        B.stride(1), B.stride(0), res.stride(1), res.stride(0));
}

// Multiplies two tensors together. Both have a rank of exactly 2.
// The second matrix is assumed to be transposed.
//
// TODO: Support multiplications like: [a, b, m, k] x [a, b, k, n] => [a, b, m, n]
// for tensors with larger ranks.
template <class T>
Tensor<T> matrixMultiplyT2(const Tensor<T>& A, const Tensor<T>& B)
{
    Tensor<T> res;
    matrixMultiplyT2(res, A, B);
    return res;
}

template <class T>
void matrixMultiplyT2(Tensor<T>& res, const Tensor<T>& A, const Tensor<T>& B)
{
    // Check size constraints
    ASSERT(A.rank() == 2 && B.rank() == 2, "A, B must be matrices");
    ASSERT(A.shape(1) == B.shape(1), "A.cols() must == B.rows(). " +
        to_string(A.shape()) + " vs. " + to_string(B.shape()));

    const size_t M = A.shape(0);
    const size_t N = B.shape(0);
    const size_t K = A.shape(1);

    res.resize({M, N});
    res.fill(T{0});
    mmtMultiply(A.data(), B.data(), res.data(), M, N, K, A.stride(1), A.stride(0),
        B.stride(1), B.stride(0), res.stride(1), res.stride(0));
}

// Calculates the L1 norm of the given tensor and returns it in
// a rank-0 tensor.
template <class T>
Tensor<T> l1Norm(const Tensor<T>& A)
{
    Tensor<T> res;
    l1Norm(res, A);
    return res;
}

// Calculates the L1 norm of the given tensor and returns it in
// a rank-0 tensor.
template <class T>
void l1Norm(Tensor<T>& y, const Tensor<T>& A)
{
    T sum{};
    for (const T& elem : A)
        sum += abs(elem);

    y.resize({1});
    y = sqrt(sum);
}

// Calculates the L2 norm of the given tensor and returns it in
// a rank-0 tensor.
template <class T>
Tensor<T> l2Norm(const Tensor<T>& A)
{
    Tensor<T> res;
    l2Norm(res, A);
    return res;
}

// Calculates the L2 norm of the given tensor and returns it in
// a rank-0 tensor.
template <class T>
void l2Norm(Tensor<T>& y, const Tensor<T>& A)
{
    T sum{};
    for (const T& elem : A)
        sum += elem * elem;

    y.resize({1});
    y = sqrt(sum);
}

// Returns the transpose of the given matrix
template <class T>
Tensor<T> matrixTranspose(const Tensor<T>& A)
{
    ASSERT(A.rank() == 2, "A must be a matrix");
    return transpose(A, 0, 1);
}

// Returns the transpose of the given matrix
template <class T>
void matrixTranspose(Tensor<T>& y, const Tensor<T>& A)
{
    ASSERT(A.rank() == 2, "A must be a matrix");
    Tensor<T> temp = transpose(A, 0, 1);
    temp.copy(y);
}

}
#endif

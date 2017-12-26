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

namespace opkit
{

// ------------------------------- Utilities -------------------------------- //

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
            to_string(aShape) + ", B = " + to_string(bShape));
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

// Apply 'f' to every individual element of the input
template <class T, class Func>
Tensor<T> elementwiseFunc(const Tensor<T>& arg, Func&& f)
{
    Tensor<T> res = arg.clone();
    res.apply(std::forward<Func>(f));
    return res;
}

// ------------------------ Broadcasting Binary Ops ------------------------- //

// Elementwise binary function that supports broadcasting so the sizes don't
// have to line up exactly.
template <class T, class Func>
Tensor<T> broadcastingBinaryOp(const Tensor<T>& A, const Tensor<T>& B, Func f)
{
    // Try to use broadcasting
    if (A.shape() != B.shape())
    {
        SmallVector commonShape = getBroadcastShape(A, B);
        return broadcastingBinaryOp(expand(A, commonShape.begin(), commonShape.end()),
            expand(B, commonShape.begin(), commonShape.end()), f);
    }

    // Apply binary op. We take advantage of the fact that 'res' is guaranteed
    // to be continuous to improve performance a bit.
    const SmallVector& shape = A.shape();
    Tensor<T> res(shape.begin(), shape.end());
    T* resData = res.data();
    auto itB   = B.begin();

    for (const T& elem : A)
    {
        *resData = f(elem, *itB);
        ++itB; ++resData;
    }

    return res;
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
    // Try to use broadcasting
    if (A.shape() != B.shape())
    {
        SmallVector commonShape = getBroadcastShape(A, B);
        return add(expand(A, commonShape.begin(), commonShape.end()),
            expand(B, commonShape.begin(), commonShape.end()));
    }

    // Apply binary op. We take advantage of the fact that 'res' is guaranteed
    // to be continuous to improve performance a bit.
    const SmallVector& shape = A.shape();
    Tensor<T> res(shape.begin(), shape.end());
    T* resData = res.data();

    if (A.contiguous() && B.contiguous())
    {
        vCopy(A.data(), resData, A.size());
        vAdd(B.data(), resData, A.size());
    }
    else if (A.rank() == 2)
    {
        // Copy A into res
        T* resDataPtr = resData;
        for (const T& elem : A)
            *resDataPtr++ = elem;

        // A += B
        matrixAdd(B.data(), resData, res.shape(0), res.shape(1),
            B.stride(1), B.stride(0), res.stride(1), res.stride(0), T{1}, T{1});
    }
    else
    {
        auto itB = B.begin();
        for (const T& elem : A)
        {
            *resData = elem + *itB;
            ++itB; ++resData;
        }
    }

    return res;
}

// Subtracts B from A. If the two tensors have the same shape, the
// subtraction is performed elementwise, as expected. If the two tensors DO NOT
// have the same shape, broadcasting will be attempted to see if the two tensors
// can possibly be coerced into having a common shape, as with add().
template <class T>
Tensor<T> sub(const Tensor<T>& A, const Tensor<T>& B)
{
    // Try to use broadcasting
    if (A.shape() != B.shape())
    {
        SmallVector commonShape = getBroadcastShape(A, B);
        return sub(expand(A, commonShape.begin(), commonShape.end()),
            expand(B, commonShape.begin(), commonShape.end()));
    }

    // Apply binary op. We take advantage of the fact that 'res' is guaranteed
    // to be continuous to improve performance a bit.
    const SmallVector& shape = A.shape();
    Tensor<T> res(shape.begin(), shape.end());
    T* resData = res.data();

    if (A.contiguous() && B.contiguous())
    {
        vCopy(A.data(), resData, A.size());
        vAdd(B.data(), resData, A.size(), T{-1});
    }
    else if (A.rank() == 2)
    {
        // Copy A into res
        T* resDataPtr = resData;
        for (const T& elem : A)
            *resDataPtr++ = elem;

        // A -= B
        matrixAdd(B.data(), resData, res.shape(0), res.shape(1),
            B.stride(1), B.stride(0), res.stride(1), res.stride(0), T{-1}, T{1});
    }
    else
    {
        auto itB = B.begin();
        for (const T& elem : A)
        {
            *resData = elem - *itB;
            ++itB; ++resData;
        }
    }

    return res;
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
    // Handle easy cases first
    if (A.size() == 1)
    {
        const T val   = T(A);
        Tensor<T> res = B.clone();
        for (T& elem : res)
            elem *= val;
        return res;
    }

    else if (B.size() == 1)
    {
        const T val   = T(B);
        Tensor<T> res = A.clone();
        for (T& elem : res)
            elem *= val;
        return res;
    }

    else return broadcastingBinaryOp(A, B, [](const T& a, const T& b)
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
    // Handle easy cases first
    if (A.size() == 1)
    {
        const T val   = T(A);
        Tensor<T> res = B.clone();
        for (T& elem : res)
            elem = val / elem;
        return res;
    }

    else if (B.size() == 1)
    {
        const T val   = T{1} / T(B);
        Tensor<T> res = A.clone();
        for (T& elem : res)
            elem *= val;
        return res;
    }

    else return broadcastingBinaryOp(A, B, [](const T& a, const T& b)
    {
        return a / b;
    });
}

template <class T>
Tensor<T> max(const Tensor<T>& A, const Tensor<T>& B)
{
    // Handle easy cases first
    if (A.size() == 1)
    {
        const T val{A};
        Tensor<T> res = B.clone();
        for (T& elem : res)
        {
            if (val > elem)
                elem = val;
        }
        return res;
    }

    else if (B.size() == 1)
    {
        const T val{B};
        Tensor<T> res = A.clone();
        for (T& elem : res)
        {
            if (val > elem)
                elem = val;
        }
        return res;
    }

    else return broadcastingBinaryOp(A, B, [](const T& a, const T& b)
    {
        return std::max(a, b);
    });
}

template <class T>
Tensor<T> min(const Tensor<T>& A, const Tensor<T>& B)
{
    // Handle easy cases first
    if (A.size() == 1)
    {
        const T val{A};
        Tensor<T> res = B.clone();
        for (T& elem : res)
        {
            if (val < elem)
                elem = val;
        }
        return res;
    }

    else if (B.size() == 1)
    {
        const T val{B};
        Tensor<T> res = A.clone();
        for (T& elem : res)
        {
            if (val < elem)
                elem = val;
        }
        return res;
    }

    else return broadcastingBinaryOp(A, B, [](const T& a, const T& b)
    {
        return std::min(a, b);
    });
}

// Clips each element of the given tensor to the range [min, max]
template <class T, class U>
Tensor<T> clip(const Tensor<T>& A, const U min, const U max)
{
    Tensor<T> res = A.clone();
    for (T& elem : res)
        elem = std::max(T{min}, std::min(elem, T{max}));
    return res;
}

// ------------------------------- Update Ops ------------------------------- //
// A += B;
template <class T>
void addTo(Tensor<T>& A, const Tensor<T>& B)
{
    // Try to use broadcasting
    if (A.shape() != B.shape())
    {
        SmallVector commonShape = getBroadcastShape(A, B);
        ASSERT(commonShape == A.shape(), "addTo: Only B's shape can change.");
        return addTo(A, expand(B, commonShape.begin(), commonShape.end()));
    }

    auto bIt = B.begin();
    for (T& elem : A)
    {
        elem += *bIt;
        ++bIt;
    }
}

// A -= B;
template <class T>
void subFrom(Tensor<T>& A, const Tensor<T>& B)
{
    // Try to use broadcasting
    if (A.shape() != B.shape())
    {
        SmallVector commonShape = getBroadcastShape(A, B);
        ASSERT(commonShape == A.shape(), "subFrom: Only B's shape can change.");
        return subFrom(A, expand(B, commonShape.begin(), commonShape.end()));
    }

    auto bIt = B.begin();
    for (T& elem : A)
    {
        elem -= *bIt;
        ++bIt;
    }
}

// A *= B;
template <class T>
void multBy(Tensor<T>& A, const Tensor<T>& B)
{
    // Try to use broadcasting
    if (A.shape() != B.shape())
    {
        SmallVector commonShape = getBroadcastShape(A, B);
        ASSERT(commonShape == A.shape(), "multBy: Only B's shape can change.");
        return multBy(A, expand(B, commonShape.begin(), commonShape.end()));
    }

    auto bIt = B.begin();
    for (T& elem : A)
    {
        elem *= *bIt;
        ++bIt;
    }
}

// A /= B;
template <class T>
void divBy(Tensor<T>& A, const Tensor<T>& B)
{
    // Try to use broadcasting
    if (A.shape() != B.shape())
    {
        SmallVector commonShape = getBroadcastShape(A, B);
        ASSERT(commonShape == A.shape(), "divBy: Only B's shape can change.");
        return divBy(A, expand(B, commonShape.begin(), commonShape.end()));
    }

    auto bIt = B.begin();
    for (T& elem : A)
    {
        elem /= *bIt;
        ++bIt;
    }
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
template <class T>
void scale(Tensor<T>& x, const T alpha)
{
    if (x.contiguous())
        vScale(x.data(), alpha, x.size());
    else
    {
        for (T& elem : x)
            elem *= alpha;
    }
}

// ---------------------------- Special Products ---------------------------- //

// Returns the inner product between the two tensors A and B. Both
// tensors must have the same shape. For vectors, this operation
// is usually called a dot product, and for matrices, it's a
// Frobenius inner product. In all cases, the result is a single
// scalar.
template <class T>
Tensor<T> innerProduct(const Tensor<T>& A, const Tensor<T>& B)
{
    ASSERT(A.shape() == B.shape(), "Shapes must match");

    T sum{};
    auto itB = B.begin();
    for (const T& elem : A)
    {
        sum += elem * *itB;
        ++itB;
    }

    return Tensor<T>::fromScalar(sum);
}

// A:      Rank 1, size N
// B:      Rank 2, size N x M
// Result: Rank 1, size N
template <class T>
Tensor<T> vectorMatrixMultiply(const Tensor<T>& A, const Tensor<T>& B)
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

// Multiplies two tensors together. Both have a rank of exactly 2.
//
// TODO: Support multiplications like: [a, b, m, k] x [a, b, k, n] => [a, b, m, n]
// for tensors with larger ranks.
template <class T>
Tensor<T> matrixMultiply(const Tensor<T>& A, const Tensor<T>& B)
{
    // Check size constraints
    ASSERT(A.rank() == 2 && B.rank() == 2, "A, B must be matrices. shape(A) = " +
        to_string(A.shape()) + ", shape(B) = " + to_string(B.shape()));
    ASSERT(A.shape(1) == B.shape(0), "A.cols() must == B.rows(). " +
        to_string(A.shape()) + " vs. " + to_string(B.shape()));

    const size_t M = A.shape(0);
    const size_t N = B.shape(1);
    const size_t K = A.shape(1);

    Tensor<T> res({M, N});
    res.fill(T{0});
    mmMultiply(A.data(), B.data(), res.data(), M, N, K, A.stride(1), A.stride(0),
        B.stride(1), B.stride(0), res.stride(1), res.stride(0));

    return res;
}

// Multiplies two tensors together. Both have a rank of exactly 2.
// The first matrix is assumed to be transposed.
//
// TODO: Support multiplications like: [a, b, m, k] x [a, b, k, n] => [a, b, m, n]
// for tensors with larger ranks.
template <class T>
Tensor<T> matrixMultiplyT1(const Tensor<T>& A, const Tensor<T>& B)
{
    // Check size constraints
    ASSERT(A.rank() == 2 && B.rank() == 2, "A, B must be matrices");
    ASSERT(A.shape(0) == B.shape(0), "A.cols() must == B.rows(). " +
        to_string(A.shape()) + " vs. " + to_string(B.shape()));

    const size_t M = A.shape(1);
    const size_t N = B.shape(1);
    const size_t K = A.shape(0);

    Tensor<T> res({M, N});
    res.fill(T{0});
    mtmMultiply(A.data(), B.data(), res.data(), M, N, K, A.stride(1), A.stride(0),
        B.stride(1), B.stride(0), res.stride(1), res.stride(0));

    return res;
}

// Multiplies two tensors together. Both have a rank of exactly 2.
// The second matrix is assumed to be transposed.
//
// TODO: Support multiplications like: [a, b, m, k] x [a, b, k, n] => [a, b, m, n]
// for tensors with larger ranks.
template <class T>
Tensor<T> matrixMultiplyT2(const Tensor<T>& A, const Tensor<T>& B)
{
    // Check size constraints
    ASSERT(A.rank() == 2 && B.rank() == 2, "A, B must be matrices");
    ASSERT(A.shape(1) == B.shape(1), "A.cols() must == B.rows(). " +
        to_string(A.shape()) + " vs. " + to_string(B.shape()));

    const size_t M = A.shape(0);
    const size_t N = B.shape(0);
    const size_t K = A.shape(1);

    Tensor<T> res({M, N});
    res.fill(T{0});
    mmtMultiply(A.data(), B.data(), res.data(), M, N, K, A.stride(1), A.stride(0),
        B.stride(1), B.stride(0), res.stride(1), res.stride(0));

    return res;
}

// Raises each element of A to the given power.
template <class T>
Tensor<T> hadamardPower(const Tensor<T>& A, const T& power)
{
    Tensor<T> res = A.clone();
    res.apply([&power](const T x)
    {
        return pow(x, power);
    });
    return res;
}

template <class T>
Tensor<T> hadamardPower(const Tensor<T>& A, const Tensor<T>& power)
{
    ASSERT(power.rank() == 0, "power must be a scalar");
    return hadamardPower(A, T(power));
}

// -------------------------------- Reduction ------------------------------- //

// Reduces the given tensor according to 'func'. When 'axes' is empty, the
// reduction is applied across the entire tensor. When non-empty, it lists
// for which axes the reduction should be performed. The resulting shape will
// match the original shape, with the exception that all axes listed in 'axes'
// will have their dimension reduced to 1.
template <class T, class Fn>
Tensor<T> reduce(const Tensor<T>& A, Fn&& func, const T& init,
    const Tensor<T>& axes)
{
    ASSERT(axes.size() <= A.rank(), "Too many axes provided.");

    SmallVector resShape(A.shape());

    // When no axes are provided, reduce on the entire tensor. Otherwise, reduce
    // only on the selected axes.
    if (axes.size() == 1 && T(axes) == -1.0)
        std::fill(resShape.begin(), resShape.end(), 1);
    else
    {
        for (const T& elem : axes)
        {
            size_t elemIndex = (size_t) elem;
            ASSERT(elemIndex < A.rank(), "Dimension axis out of bounds.");
            resShape[elemIndex] = 1;
        }
    }
    return reduceTo(A, std::forward<Fn>(func), init, resShape);
}

// Reduces 'A' to the given shape rather than specifying which indices to reduce
// on. Allowable shapes are those for which the dimensions either match exactly
// to the original shape or are replaced with a 1. Fewer dimensions are allowed,
// but this function cannot be used to add additional dimensions.
//
// Example:
// A.shape() = [3, 2]
// reduceTo( [1, 2] )    - OK
// reduceTo( [3, 1] )    - OK
// reduceTo( [1, 1] )    - OK
// reduceTo( [1] )       - OK
// reduceTo( [2] )       - OK
// reduceTo( [] )        - OK
// reduceTo( [1, 3] )    - NOT OK - Dimensions don't match
// reduceTo( [1, 3, 2] ) - NOT OK - Can't add dimensions
template <class T, class Fn, class Alloc>
Tensor<T> reduceTo(const Tensor<T>& A, Fn&& func, const T& init,
    const std::vector<size_t, Alloc>& resShape)
{
    // No reduction to be done if the shape already matches
    if (resShape == A.shape())
        return A;

    // Check to ensure the sizes line up correctly
    #ifndef NDEBUG
    {
        const SmallVector& origShape = A.shape();
        int srcIndex                 = A.rank() - 1;
        int resIndex                 = resShape.size() - 1;
        for (; resIndex >= 0 && srcIndex >= 0; --resIndex, --srcIndex)
        {
            ASSERT(resShape[resIndex] == origShape[srcIndex] ||
                resShape[resIndex] == 1,
                "reduceTo(): Sizes not compatable. " + to_string(origShape) +
                " -> " + to_string(resShape));
        }
        ASSERT(resIndex < 0, "reduceTo() cannot add dimensions. A.shape() = " +
            to_string(A.shape()) + ", resShape = " + to_string(resShape));
    }
    #endif

    Tensor<T> res(resShape.begin(), resShape.end());
    res.fill(init);

    const size_t N               = A.rank();
    const size_t M               = resShape.size();
    const size_t offset          = N - M;
    const SmallVector& aShape    = A.shape();
    const SmallVector& resStride = res.stride();
    auto end                     = A.end();
    SmallVector index(N);

    // All members of A will participate in the reduction
    for (auto it = A.begin(); it != end; ++it)
    {
        // Work out where we are in 'res' based on the current index
        T* data = res.data();
        for (size_t i = 0; i < M; ++i)
        {
            if (index[offset + i] < resShape[i])
                data += index[offset + i] * resStride[i];
        }

        // Apply the reduction
        *data = func(*data, *it);

        // Update the index (N-ary counter)
        int dim = N - 1;
        index[dim]++;
        while (dim > 0 && index[dim] >= aShape[dim])
        {
            index[dim] = 0;
            index[dim - 1]++;
            --dim;
        }
    }

    return res;
}

// Convenience overload for the graph methods that must work with tensors.
template <class T, class Fn>
Tensor<T> reduceTo(const Tensor<T>& A, Fn&& func, const T& init, const Tensor<T> shape)
{
    SmallVector resShape(shape.size());
    auto resIt = resShape.begin();
    for (const T& elem : shape)
    {
        *resIt = (size_t) elem;
        ++resIt;
    }

    return reduceTo(A, std::forward<Fn>(func), init, resShape);
}

// Reduces the given tensor using summation.
template <class T>
Tensor<T> reduceSum(const Tensor<T>& A)
{
    T sum = T{};
    if (A.contiguous())
    {
        const T* aData      = A.data();
        const size_t length = A.size();
        for (size_t i = 0; i < length; ++i)
            sum += aData[i];
        return Tensor<T>::fromScalar(sum);
    }
    else
    {
        for (const T& elem : A)
            sum += elem;
        return Tensor<T>::fromScalar(sum);
    }
}

// Reduces the given tensor using summation.
template <class T>
Tensor<T> reduceSum(const Tensor<T>& A, const Tensor<T>& axes)
{
    return reduce(A, [](const T& a, const T& b) { return a + b; }, T{}, axes);
}

// Reduces the given tensor to the given shape using summation. When the desired
// shape is {1}, the entire tensor is reduced.
template <class T>
Tensor<T> reduceSumTo(const Tensor<T>& A, const Tensor<T>& shape)
{
    // Catch easy case of complete reduction
    if (shape.size() == 1 && T(shape) == T{1})
        return reduceSum(A);
    else return reduceTo(A, [](const T& a, const T& b) { return a + b; }, T{}, shape);
}

// Reduces the given tensor using multiplication.
template <class T>
Tensor<T> reduceProduct(const Tensor<T>& A)
{
    T product = T{1};
    if (A.contiguous())
    {
        const T* aData      = A.data();
        const size_t length = A.size();
        for (size_t i = 0; i < length; ++i)
            product *= aData[i];
        return Tensor<T>::fromScalar(product);
    }
    else
    {
        for (const T& elem : A)
            product *= elem;
        return Tensor<T>::fromScalar(product);
    }
}

// Reduces the given tensor using multiplication.
template <class T>
Tensor<T> reduceProduct(const Tensor<T>& A, const Tensor<T>& axes)
{
    return reduce(A, [](const T& a, const T& b) { return a * b; }, T{1}, axes);
}

// Reduces the given tensor to the given shape using multiplication. When the desired
// shape is {1}, the entire tensor is reduced.
template <class T>
Tensor<T> reduceProductTo(const Tensor<T>& A, const Tensor<T>& shape)
{
    // Catch easy case of complete reduction
    if (shape.size() == 1 && T(shape) == T{1})
        return reduceProduct(A);
    else return reduceTo(A, [](const T& a, const T& b) { return a * b; }, T{1}, shape);
}

// Reduces the given tensor using min.
template <class T>
Tensor<T> reduceMin(const Tensor<T>& A)
{
    T min = std::numeric_limits<T>::max();
    if (A.contiguous())
    {
        const T* aData      = A.data();
        const size_t length = A.size();
        for (size_t i = 0; i < length; ++i)
        {
            if (aData[i] < min)
                min = aData[i];
        }
        return Tensor<T>::fromScalar(min);
    }
    else
    {
        for (const T& elem : A)
        {
            if (elem < min)
                min = elem;
        }
        return Tensor<T>::fromScalar(min);
    }
}

// Reduces the given tensor using min.
template <class T>
Tensor<T> reduceMin(const Tensor<T>& A, const Tensor<T>& axes)
{
    return reduce(A, [](const T& a, const T& b) { return std::min(a, b); },
        std::numeric_limits<T>::max(), axes);
}

// Reduces the given tensor to the given shape using min. When the desired
// shape is {1}, the entire tensor is reduced.
template <class T>
Tensor<T> reduceMinTo(const Tensor<T>& A, const Tensor<T>& shape)
{
    // Catch easy case of complete reduction
    if (shape.size() == 1 && T(shape) == T{1})
        return reduceMin(A);
    else return reduceTo(A, [](const T& a, const T& b) { return std::min(a, b); },
        std::numeric_limits<T>::max(), shape);
}

// Reduces the given tensor using min.
template <class T>
Tensor<T> reduceMax(const Tensor<T>& A)
{
    T max = std::numeric_limits<T>::lowest();
    if (A.contiguous())
    {
        const T* aData      = A.data();
        const size_t length = A.size();
        for (size_t i = 0; i < length; ++i)
        {
            if (aData[i] > max)
                max = aData[i];
        }
        return Tensor<T>::fromScalar(max);
    }
    else
    {
        for (const T& elem : A)
        {
            if (elem > max)
                max = elem;
        }
        return Tensor<T>::fromScalar(max);
    }
}

// Reduces the given tensor using max.
template <class T>
Tensor<T> reduceMax(const Tensor<T>& A, const Tensor<T>& axes)
{
    return reduce(A, [](const T& a, const T& b) { return std::max(a, b); },
        std::numeric_limits<T>::lowest(), axes);
}

// Reduces the given tensor to the given shape using max. When the desired
// shape is {1}, the entire tensor is reduced.
template <class T>
Tensor<T> reduceMaxTo(const Tensor<T>& A, const Tensor<T>& shape)
{
    // Catch easy case of complete reduction
    if (shape.size() == 1 && T(shape) == T{1})
        return reduceMax(A);
    else return reduceTo(A, [](const T& a, const T& b) { return std::max(a, b); },
        std::numeric_limits<T>::lowest(), shape);
}

// Reduces the given tensor using mean.
template <class T>
Tensor<T> reduceMean(const Tensor<T>& A)
{
    Tensor<T> res = reduceSum(A);
    res.at({0})  /= A.size();
    return res;
}

// Reduces the given tensor using mean.
template <class T>
Tensor<T> reduceMean(const Tensor<T>& A, const Tensor<T>& axes)
{
    // Calculate the sum reduction
    Tensor<T> res = reduceSum(A, axes);

    // Divide by the proper constant
    T constant = 1;
    if (axes.size() == 1 && T(axes) == -1.0)
        constant = A.size();
    else
    {
        for (const T& elem : axes)
            constant *= A.shape(elem);
    }
    scale(res, T{1} / constant);
    return res;
}

// Reduces the given tensor to the given shape using mean. When the desired
// shape is empty, the entire tensor is reduced.
template <class T>
Tensor<T> reduceMeanTo(const Tensor<T>& A, const Tensor<T>& shape)
{
    // Calculate the sum reduction
    Tensor<T> res = reduceSumTo(A, shape);

    ASSERT(false, "Not implemented");
    // Divide by the proper constant
    // T constant = 1;
    // if (axes.size() == 0)
    //     constant = A.size();
    // else
    // {
    //     for (auto& elem : axes)
    //         constant *= A.shape(elem);
    // }
    // res.apply([&constant](const T& x) { return x / constant; });
    return res;
}

// Calculates the index of the largest value along the given dimension. In the
// event of a tie, the index of the first of the largest values will be returned.
template <class T>
Tensor<T> argmax(const Tensor<T>& A, const size_t dimension)
{
    ASSERT(dimension < A.rank(), "Invalid dimension provided.");

    // Calculate the new shape
    SmallVector resShape(A.shape());
    resShape[dimension] = 1;

    Tensor<T> res(resShape.begin(), resShape.end());
    res.fill(T{});

    Tensor<T> max(resShape.begin(), resShape.end());
    max.fill(std::numeric_limits<T>::lowest());

    size_t N                     = A.rank();
    const SmallVector& aShape    = A.shape();
    const SmallVector& resStride = res.stride();
    auto end                     = A.end();
    SmallVector index(N);

    // All members of A will participate in the reduction
    for (auto it = A.begin(); it != end; ++it)
    {
        // Work out where we are in 'res' based on the current index
        T* resData = res.data();
        for (size_t i = 0; i < N; ++i)
        {
            if (index[i] < resShape[i])
                resData += index[i] * resStride[i];
        }

        // We should be at the same spot in 'max'
        T* maxData = max.data() + (resData - res.data());

        // Apply the reduction
        if (*it > *maxData)
        {
            *maxData = *it;
            *resData = index[dimension];
        }

        // Update the index (N-ary counter)
        int dim = N - 1;
        index[dim]++;
        while (dim > 0 && index[dim] >= aShape[dim])
        {
            index[dim] = 0;
            index[dim - 1]++;
            --dim;
        }
    }

    return res;
}

// Calculates the index of the snallest value along the given dimension. In the
// event of a tie, the index of the first of the smallest values will be returned.
template <class T>
Tensor<T> argmin(const Tensor<T>& A, const size_t dimension)
{
    ASSERT(dimension < A.rank(), "Invalid dimension provided.");

    // Calculate the new shape
    SmallVector resShape(A.shape());
    resShape[dimension] = 1;

    Tensor<T> res(resShape.begin(), resShape.end());
    res.fill(T{});

    Tensor<T> min(resShape.begin(), resShape.end());
    min.fill(std::numeric_limits<T>::max());

    size_t N                     = A.rank();
    const SmallVector& aShape    = A.shape();
    const SmallVector& resStride = res.stride();
    auto end                     = A.end();
    SmallVector index(N);

    // All members of A will participate in the reduction
    for (auto it = A.begin(); it != end; ++it)
    {
        // Work out where we are in 'res' based on the current index
        T* resData = res.data();
        for (size_t i = 0; i < N; ++i)
        {
            if (index[i] < resShape[i])
                resData += index[i] * resStride[i];
        }

        // We should be at the same spot in 'min'
        T* minData = min.data() + (resData - res.data());

        // Apply the reduction
        if (*it < *minData)
        {
            *minData = *it;
            *resData = index[dimension];
        }

        // Update the index (N-ary counter)
        int dim = N - 1;
        index[dim]++;
        while (dim > 0 && index[dim] >= aShape[dim])
        {
            index[dim] = 0;
            index[dim - 1]++;
            --dim;
        }
    }

    return res;
}

// --------------------------- Boolean Ops --------------------------- //

template <class T>
Tensor<T> equal(const Tensor<T>& A, const Tensor<T>& B)
{
    return broadcastingBinaryOp(A, B, [](const T& a, const T& b)
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
Tensor<T> greater(const Tensor<T>& A, const Tensor<T>& B)
{
    return broadcastingBinaryOp(A, B, [](const T& a, const T& b)
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
Tensor<T> less(const Tensor<T>& A, const Tensor<T>& B)
{
    return broadcastingBinaryOp(A, B, [](const T& a, const T& b)
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
// --------------------------- [Category] --------------------------- //

// Calculates the L1 norm of the given tensor and returns it in
// a rank-0 tensor.
template <class T>
Tensor<T> l1Norm(const Tensor<T>& A)
{
    T sum{};
    for (const T& elem : A)
        sum += abs(elem);

    return Tensor<T>::fromScalar(sqrt(sum));
}

// Calculates the L2 norm of the given tensor and returns it in
// a rank-0 tensor.
template <class T>
Tensor<T> l2Norm(const Tensor<T>& A)
{
    T sum{};
    for (const T& elem : A)
        sum += elem * elem;

    return Tensor<T>::fromScalar(sqrt(sum));
}

// Returns the transpose of the given matrix
template <class T>
Tensor<T> matrixTranspose(const Tensor<T>& A)
{
    ASSERT(A.rank() == 2, "A must be a matrix");
    return transpose(A, 0, 1);
}

}

#endif

#ifndef TENSOR_REDUCTIONS_H
#define TENSOR_REDUCTIONS_H

#include "acceleration/Acceleration.h"
#include "tensor/Tensor.h"
#include "tensor/TensorMath.h"

namespace detail
{

// Helper for reduceTo() that uses a routine optimized for reducing matrices.
template <class T, class Fn>
void reduceTo2D(Tensor<T>& res, const Tensor<T>& A, Fn&& func, const T& init,
    const size_t M, const size_t N)
{
    res.resize({M, N});

    const size_t xStride = A.stride(1);
    const size_t yStride = A.stride(0);
    const size_t width   = A.shape(1);
    const size_t height  = A.shape(0);

    const T* dataPtr = A.data();
          T* resPtr  = res.data();

    // Reduce along dimension 0
    if (M == 1)
    {
        res.fill(init);
        for (size_t y = 0; y < height; ++y)
        {
            const T* data = dataPtr;
            for (size_t x = 0; x < width; ++x)
            {
                resPtr[x] = func(resPtr[x], *data);
                data += xStride;
            }
            dataPtr += yStride;
        }
    }

    // Reduce along dimension 1
    else
    {
        for (size_t y = 0; y < height; ++y)
        {
            const T* data = dataPtr;
            T val         = init;
            for (size_t x = 0; x < width; ++x)
            {
                val   = func(val, *data);
                data += xStride;
            }
            resPtr[y] = val;
            dataPtr  += yStride;
        }
    }
}

// Helper for reduceTo() that uses a generalized routine suitable for arbitrary
// tensors.
template <class T, class Fn, class Alloc>
void reduceToGeneral(Tensor<T>& res, const Tensor<T>& A, Fn&& func, const T& init,
    const std::vector<size_t, Alloc>& resShape)
{
    res.resize(resShape.begin(), resShape.end());
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
}

// Helper for argmin()/argmax() that uses a routine optimized for reducing
// matrices. AccelFunc should either be opkit::vMaxIndex or opkit::vMinIndex.
template <class T, class AccelFunc>
void reduceArg2D(Tensor<T>& res, const Tensor<T>& A, const size_t dimension, AccelFunc&& f)
{
    const size_t M = A.shape(0);
    const size_t N = A.shape(1);

    // Calculate the new shape
    SmallVector resShape(A.shape());
    resShape[dimension] = 1;
    res.resize(resShape.begin(), resShape.end());

    // Flattening columns (most common case)
    if (dimension == 1)
    {
        //vMaxIndex(const T* x, const size_t N, const int xInc = 1)
        const size_t aStrideX = A.stride(1);
        const size_t aStrideY = A.stride(0);
        const T* aData        = A.data();
              T* resData      = res.data();

        for (size_t y = 0; y < M; ++y)
        {
            resData[y] = f(aData, N, aStrideX);
            aData     += aStrideY;
        }
    }

    // Flattening rows (less common case)
    else
    {
        const size_t aStrideX = A.stride(1);
        const size_t aStrideY = A.stride(0);
        const T* aData        = A.data();
              T* resData      = res.data();

        for (size_t x = 0; x < N; ++x)
        {
            resData[x] = f(aData, M, aStrideY);
            aData     += aStrideX;
        }
    }
}

// Helper for argmin()/argmax() that uses a generalized routine suitable for
// arbitrary tensors. The comparator is used to differentiate between min and
// max. 'initValue' should be a large positive number for min() and a large
// negative number for max.
template <class T, class Comparator>
void reduceArgGeneral(Tensor<T>& res, const Tensor<T>& A, const size_t dimension,
    const T initValue, Comparator&& comp)
{
    // Calculate the new shape
    SmallVector resShape(A.shape());
    resShape[dimension] = 1;

    res.resize(resShape.begin(), resShape.end());
    res.fill(T{});

    Tensor<T> min(resShape.begin(), resShape.end());
    min.fill(initValue);

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
        if (comp(*it, *minData))
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
}

}

namespace opkit
{

// --------------------------- General Reduction ---------------------------- //

// Reduces the given tensor according to 'func'. When 'axes' is -1, the
// reduction is applied across the entire tensor. Otherwise, it lists
// for which axes the reduction should be performed. The resulting shape will
// match the original shape, with the exception that all axes listed in 'axes'
// will have their dimension reduced to 1.
template <class T, class Fn, class Alloc>
Tensor<T> reduce(const Tensor<T>& A, Fn&& func, const T& init,
    const vector<size_t, Alloc>& axes)
{
    Tensor<T> res;
    reduce(res, A, std::forward<Fn>(func), init, axes);
    return res;
}

template <class T, class Fn, class Alloc>
void reduce(Tensor<T>& y, const Tensor<T>& A, Fn&& func, const T& init,
    const vector<size_t, Alloc>& axes)
{
    ASSERT(axes.size() <= A.rank(), "Too many axes provided.");

    SmallVector resShape(A.shape());

    // When no axes are provided, reduce on the entire tensor. Otherwise, reduce
    // only on the selected axes.
    if (axes.size() == 1 && axes[0] == -1)
        std::fill(resShape.begin(), resShape.end(), 1);
    else
    {
        for (const size_t& elem : axes)
        {
            ASSERT(elem < A.rank(), "Dimension axis out of bounds.");
            resShape[elem] = 1;
        }
    }
    reduceTo(y, A, std::forward<Fn>(func), init, resShape);
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
    Tensor<T> res;
    reduceTo(res, A, std::forward<Fn>(func), init, resShape);
    return res;
}

template <class T, class Fn, class Alloc>
void reduceTo(Tensor<T>& y, const Tensor<T>& A, Fn&& func, const T& init,
    const std::vector<size_t, Alloc>& resShape)
{
    // No reduction to be done if the shape already matches
    if (resShape == A.shape())
    {
        A.copy(y);
        return;
    }

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

    // Apply an optimization for 2D tensors
    if (A.rank() == 2)
        detail::reduceTo2D(y, A, std::forward<Fn>(func), init, resShape[0], resShape[1]);

    // Use the more expensive but general routine otherwise
    else detail::reduceToGeneral(y, A, std::forward<Fn>(func), init, resShape);
}

// Convenience overload for the graph methods that must work with tensors.
template <class T, class Fn>
Tensor<T> reduceTo(const Tensor<T>& A, Fn&& func, const T& init, const Tensor<T> shape)
{
    Tensor<T> res;
    reduceTo(res, A, std::forward<Fn>(func), init, shape);
    return res;
}

template <class T, class Fn>
void reduceTo(Tensor<T>& y, const Tensor<T>& A, Fn&& func, const T& init, const Tensor<T> shape)
{
    SmallVector resShape(shape.size());
    auto resIt = resShape.begin();
    for (const T& elem : shape)
    {
        *resIt = (size_t) elem;
        ++resIt;
    }

    reduceTo(y, A, std::forward<Fn>(func), init, resShape);
}

// -------------------------- Specific Reductions --------------------------- //

// Reduces the given tensor using summation.
template <class T>
Tensor<T> reduceSum(const Tensor<T>& A)
{
    Tensor<T> res;
    reduceSum(res, A);
    return res;
}

template <class T>
void reduceSum(Tensor<T>& y, const Tensor<T>& A)
{
    T sum = T{};
    if (A.contiguous())
    {
        const T* aData      = A.data();
        const size_t length = A.size();
        for (size_t i = 0; i < length; ++i)
            sum += aData[i];
        y.resize({1});
        y = sum;
    }
    else
    {
        for (const T& elem : A)
            sum += elem;
        y.resize({1});
        y = sum;
    }
}

// Reduces the given tensor using summation.
template <class T, class Alloc>
Tensor<T> reduceSum(const Tensor<T>& A, const vector<size_t, Alloc>& axes)
{
    return reduce(A, [](const T& a, const T& b) { return a + b; }, T{}, axes);
}

template <class T, class Alloc>
void reduceSum(Tensor<T>& y, const Tensor<T>& A, const vector<size_t, Alloc>& axes)
{
    reduce(y, A, [](const T& a, const T& b) { return a + b; }, T{}, axes);
}

// Reduces the given tensor to the given shape using summation. When the desired
// shape is {1}, the entire tensor is reduced.
template <class T>
Tensor<T> reduceSumTo(const Tensor<T>& A, const Tensor<T>& shape)
{
    Tensor<T> res;
    reduceSumTo(res, A, shape);
    return res;
}

// Reduces the given tensor to the given shape using summation. When the desired
// shape is {1}, the entire tensor is reduced.
template <class T>
void reduceSumTo(Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& shape)
{
    // Catch easy case of complete reduction
    if (shape.size() == 1 && T(shape) == T{1})
        reduceSum(y, A);
    else reduceTo(y, A, [](const T& a, const T& b) { return a + b; }, T{}, shape);
}


// Reduces the given tensor using multiplication.
template <class T>
Tensor<T> reduceProduct(const Tensor<T>& A)
{
    Tensor<T> res;
    reduceProduct(res, A);
    return res;
}

template <class T>
void reduceProduct(Tensor<T>& y, const Tensor<T>& A)
{
    T product = T{1};
    if (A.contiguous())
    {
        const T* aData      = A.data();
        const size_t length = A.size();
        for (size_t i = 0; i < length; ++i)
            product *= aData[i];
        y.resize({1});
        y = product;
    }
    else
    {
        for (const T& elem : A)
            product *= elem;
        y.resize({1});
        y = product;
    }
}

// Reduces the given tensor using multiplication.
template <class T, class Alloc>
Tensor<T> reduceProduct(const Tensor<T>& A, const vector<size_t, Alloc>& axes)
{
    return reduce(A, [](const T& a, const T& b) { return a * b; }, T{1}, axes);
}

template <class T, class Alloc>
void reduceProduct(Tensor<T>& y, const Tensor<T>& A, const vector<size_t, Alloc>& axes)
{
    reduce(y, A, [](const T& a, const T& b) { return a * b; }, T{1}, axes);
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

template <class T>
void reduceProductTo(Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& shape)
{
    // Catch easy case of complete reduction
    if (shape.size() == 1 && T(shape) == T{1})
        reduceProduct(y, A);
    else reduceTo(y, A, [](const T& a, const T& b) { return a * b; }, T{1}, shape);
}

// Reduces the given tensor using min.
template <class T>
Tensor<T> reduceMin(const Tensor<T>& A)
{
    Tensor<T> res;
    reduceMin(res, A);
    return res;
}

template <class T>
void reduceMin(Tensor<T>& y, const Tensor<T>& A)
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
        y.resize({1});
        y = min;
    }
    else
    {
        for (const T& elem : A)
        {
            if (elem < min)
                min = elem;
        }
        y.resize({1});
        y = min;
    }
}

// Reduces the given tensor using min.
template <class T, class Alloc>
Tensor<T> reduceMin(const Tensor<T>& A, const vector<size_t, Alloc>& axes)
{
    return reduce(A, [](const T& a, const T& b) { return std::min(a, b); },
        std::numeric_limits<T>::max(), axes);
}

// Reduces the given tensor using min.
template <class T, class Alloc>
void reduceMin(Tensor<T>& y, const Tensor<T>& A, const vector<size_t, Alloc>& axes)
{
    reduce(y, A, [](const T& a, const T& b) { return std::min(a, b); },
        std::numeric_limits<T>::max(), axes);
}

// Reduces the given tensor to the given shape using min. When the desired
// shape is {1}, the entire tensor is reduced.
template <class T>
Tensor<T> reduceMinTo(const Tensor<T>& A, const Tensor<T>& shape)
{
    Tensor<T> res;
    reduceMinTo(res, A, shape);
    return res;
}

template <class T>
void reduceMinTo(Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& shape)
{
    // Catch easy case of complete reduction
    if (shape.size() == 1 && T(shape) == T{1})
        reduceMin(y, A);
    else reduceTo(y, A, [](const T& a, const T& b) { return std::min(a, b); },
        std::numeric_limits<T>::max(), shape);
}


// Reduces the given tensor using max.
template <class T>
Tensor<T> reduceMax(const Tensor<T>& A)
{
    Tensor<T> res;
    reduceMax(res, A);
    return res;
}

template <class T>
void reduceMax(Tensor<T>& y, const Tensor<T>& A)
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
        y.resize({1});
        y = max;
    }
    else
    {
        for (const T& elem : A)
        {
            if (elem > max)
                max = elem;
        }
        y.resize({1});
        y = max;
    }
}

// Reduces the given tensor using max.
template <class T, class Alloc>
Tensor<T> reduceMax(const Tensor<T>& A, const vector<size_t, Alloc>& axes)
{
    return reduce(A, [](const T& a, const T& b) { return std::max(a, b); },
        std::numeric_limits<T>::lowest(), axes);
}

template <class T, class Alloc>
void reduceMax(Tensor<T>& y, const Tensor<T>& A, const vector<size_t, Alloc>& axes)
{
    reduce(y, A, [](const T& a, const T& b) { return std::max(a, b); },
        std::numeric_limits<T>::lowest(), axes);
}

// Reduces the given tensor to the given shape using max. When the desired
// shape is {1}, the entire tensor is reduced.
template <class T>
Tensor<T> reduceMaxTo(const Tensor<T>& A, const Tensor<T>& shape)
{
    Tensor<T> res;
    reduceMaxTo(res, A, shape);
    return res;
}

template <class T>
void reduceMaxTo(Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& shape)
{
    // Catch easy case of complete reduction
    if (shape.size() == 1 && T(shape) == T{1})
        reduceMax(y, A);
    else reduceTo(y, A, [](const T& a, const T& b) { return std::max(a, b); },
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

template <class T>
void reduceMean(Tensor<T>& y, const Tensor<T>& A)
{
    reduceSum(y, A);
    y.at({0}) /= A.size();
}

// Reduces the given tensor using mean.
template <class T, class Alloc>
Tensor<T> reduceMean(const Tensor<T>& A, const vector<size_t, Alloc>& axes)
{
    Tensor<T> res;
    reduceMean(res, A, axes);
    return res;
}

template <class T, class Alloc>
void reduceMean(Tensor<T>& y, const Tensor<T>& A, const vector<size_t, Alloc>& axes)
{
    // Calculate the sum reduction
    reduceSum(y, A, axes);

    // Divide by the proper constant
    T constant = 1;
    if (axes.size() == 1 && axes[0] == -1)
        constant = A.size();
    else
    {
        for (const T& elem : axes)
            constant *= A.shape(elem);
    }
    scale(y, T{1} / constant);
}

// Reduces the given tensor to the given shape using mean. When the desired
// shape is empty, the entire tensor is reduced.
template <class T>
Tensor<T> reduceMeanTo(const Tensor<T>& A, const Tensor<T>& shape)
{
    Tensor<T> res;
    reduceMeanTo(res, A, shape);
    return res;
}

template <class T>
void reduceMeanTo(Tensor<T>& y, const Tensor<T>& A, const Tensor<T>& shape)
{
    // Calculate the sum reduction
    reduceSumTo(y, A, shape);
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
}

// Calculates the index of the largest value along the given dimension. In the
// event of a tie, the index of the first of the largest values will be returned.
template <class T>
Tensor<T> argmax(const Tensor<T>& A, const size_t dimension)
{
    Tensor<T> res;
    argmax(res, A, dimension);
    return res;
}

template <class T>
void argmax(Tensor<T>& y, const Tensor<T>& A, const size_t dimension)
{
    ASSERT(dimension < A.rank(), "Invalid dimension provided.");

    // Choose the better implementation depending on A
    if (A.rank() == 2)
        detail::reduceArg2D(y, A, dimension,
        [](const T* x, const size_t N, const int xInc)
        {
            return vMaxIndex(x, N, xInc);
        });

    else detail::reduceArgGeneral(y, A, dimension,
        std::numeric_limits<T>::lowest(), std::greater<T>());
}

// Calculates the index of the snallest value along the given dimension. In the
// event of a tie, the index of the first of the smallest values will be returned.
template <class T>
Tensor<T> argmin(const Tensor<T>& A, const size_t dimension)
{
    Tensor<T> res;
    argmin(res, A, dimension);
    return res;
}

template <class T>
void argmin(Tensor<T>& y, const Tensor<T>& A, const size_t dimension)
{
    ASSERT(dimension < A.rank(), "Invalid dimension provided.");

    // Choose the better implementation depending on A
    if (A.rank() == 2)
        detail::reduceArg2D(y, A, dimension,
        [](const T* x, const size_t N, const int xInc)
        {
            return vMinIndex(x, N, xInc);
        });

    else detail::reduceArgGeneral(y, A, dimension,
        std::numeric_limits<T>::max(), std::less<T>());
}

}
#endif

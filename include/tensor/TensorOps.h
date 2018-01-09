#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include <chrono>
#include "util/Rand.h"

namespace opkit
{

// ------------------------------- Generators ------------------------------- //

// Returns a new tensor with the given shape whose values are initialized to a
// range from [start, N], where N is calculated based on the shape. 'inc' can
// be used to adjust the increment amount (e.g. [0, 1, 2] vs. [0, 0.5, 1.0]).
template <class T>
Tensor<T> range(std::initializer_list<size_t> shape,
    const T start = T{}, const T inc = T{1})
{
    Tensor<T> res(shape);
    Storage<T>& storage = res.storage();

    T val = start;
    for (T& elem : storage)
    {
        elem = val;
        val += inc;
    }
    return res;
}

template <class T, class U>
Tensor<T> rangeLike(const Tensor<U>& model, const T start = T{}, const T inc = T{1})
{
    const SmallVector& shape = model.shape();
    Tensor<T> res(shape.begin(), shape.end());
    Storage<T>& storage = res.storage();

    T val = start;
    for (T& elem : storage)
    {
        elem = val;
        val += inc;
    }
    return res;
}

// Returns a new tensor with the given shape whose values are all initialized
// to 0.0.
template <class T>
Tensor<T> zeroes(std::initializer_list<size_t> shape)
{
    Tensor<T> res(shape);
    res.fill(T{});
    return res;
}

template <class T, class Alloc>
Tensor<T> zeroes(const std::vector<T, Alloc>& shape)
{
    Tensor<T> res(shape.begin(), shape.end());
    res.fill(T{});
    return res;
}

template <class T, class U>
Tensor<T> zeroesLike(const Tensor<U>& model)
{
    const SmallVector& shape = model.shape();
    Tensor<T> res(shape.begin(), shape.end());
    res.fill(T{});
    return res;
}

// Returns a new tensor with the given shape whose values are all initialized
// to 1.0.
template <class T>
Tensor<T> ones(std::initializer_list<size_t> shape)
{
    Tensor<T> res(shape);
    res.fill(T{1});
    return res;
}

template <class T, class Alloc>
Tensor<T> ones(const std::vector<size_t, Alloc>& shape)
{
    Tensor<T> res(shape.begin(), shape.end());
    res.fill(T{1});
    return res;
}

template <class T, class U>
Tensor<T> onesLike(const Tensor<U>& model)
{
    const SmallVector& shape = model.shape();
    Tensor<T> res(shape.begin(), shape.end());
    res.fill(T{1});
    return res;
}

template <class T>
Tensor<T> identity(const size_t rows, const size_t cols)
{
    Tensor<T> res({rows, cols});
    res.fill(T{});
    for (size_t i = 0; i < std::min(rows, cols); ++i)
        res.at({i, i}) = T{1};
    return res;
}

// Returns a new tensor with the given shape whose values are initialized
// according to a uniform distribution.
template <class T>
Tensor<T> uniform(std::initializer_list<size_t> shape,
    const T start = T{}, const T end = T{1},
    const size_t seed = std::chrono::system_clock::now().time_since_epoch().count(),
    typename std::enable_if<std::is_floating_point<T>::value >::type* = 0)
{
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<T> distribution(start, end);

    Tensor<T> res(shape);
    for (T& elem : res)
        elem = distribution(generator);
    return res;
}

template <class T>
Tensor<T> uniform(std::initializer_list<size_t> shape, Rand& rand,
    const T start = T{}, const T end = T{1},
    typename std::enable_if<std::is_floating_point<T>::value >::type* = 0)
{
    Tensor<T> res(shape);
    for (T& elem : res)
        elem = rand.nextReal(start, end);
    return res;
}

template <class T, class U>
Tensor<T> uniformLike(const Tensor<U>& model, Rand& rand,
    const T start = T{}, const T end = T{1},
    typename std::enable_if<std::is_floating_point<T>::value >::type* = 0)
{
    const SmallVector& shape = model.shape();
    Tensor<T> res(shape.begin(), shape.end());
    for (T& elem : res)
        elem = rand.nextReal(start, end);
    return res;
}

template <class T>
Tensor<T> uniform(std::initializer_list<size_t> shape,
    const T start = T{}, const T end = T{9},
    const size_t seed = std::chrono::system_clock::now().time_since_epoch().count(),
    typename std::enable_if<std::is_integral<T>::value >::type* = 0)
{
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<T> distribution(start, end);

    Tensor<T> res(shape);
    for (T& elem : res)
        elem = distribution(generator);
    return res;
}

template <class T>
Tensor<T> uniform(std::initializer_list<size_t> shape, Rand& rand,
    const T start = T{}, const T end = T{9},
    typename std::enable_if<std::is_integral<T>::value >::type* = 0)
{
    Tensor<T> res(shape);
    for (T& elem : res)
        elem = rand.nextInteger(start, end);
    return res;
}

template <class T, class U>
Tensor<T> uniformLike(const Tensor<U>& model, Rand& rand,
    const T start = T{}, const T end = T{9},
    typename std::enable_if<std::is_integral<T>::value >::type* = 0)
{
    const SmallVector& shape = model.shape();
    Tensor<T> res(shape.begin(), shape.end());
    for (T& elem : res)
        elem = rand.nextInteger(start, end);
    return res;
}

// Returns a new tensor with the given shape whose values are initialized
// according to a normal distribution.
template <class T>
Tensor<T> normal(std::initializer_list<size_t> shape,
    const T mean = T{}, const T stdev = T{1},
    const size_t seed = std::chrono::system_clock::now().time_since_epoch().count(),
    typename std::enable_if<std::is_floating_point<T>::value >::type* = 0)
{
    std::default_random_engine generator(seed);
    std::normal_distribution<T> distribution(mean, stdev);

    Tensor<T> res(shape);
    for (T& elem : res)
        elem = distribution(generator);
    return res;
}

template <class T>
Tensor<T> normal(std::initializer_list<size_t> shape, Rand& rand,
    const T mean = T{}, const T stdev = T{1},
    typename std::enable_if<std::is_floating_point<T>::value >::type* = 0)
{
    Tensor<T> res(shape);
    for (T& elem : res)
        elem = rand.nextGaussian(mean, stdev);
    return res;
}

template <class T, class U>
Tensor<T> normalLike(const Tensor<U>& model, Rand& rand,
    const T mean = T{}, const T stdev = T{1},
    typename std::enable_if<std::is_floating_point<T>::value >::type* = 0)
{
    const SmallVector& shape = model.shape();
    Tensor<T> res(shape.begin(), shape.end());
    for (T& elem : res)
        elem = rand.nextGaussian(mean, stdev);
    return res;
}

template <class T>
Tensor<T> xavier(std::initializer_list<size_t> shape, Rand& rand)
{
    T stdev = sqrt(T{2} / shape[0]);
    return normal<T>(shape, rand, 0, stdev);
}

template <class T, class U>
Tensor<T> xavierLike(const Tensor<U>& model, Rand& rand)
{
    T stdev = sqrt(T{2} / model.shape(0));

    const SmallVector& shape = model.shape();
    Tensor<T> res(shape.begin(), shape.end());
    for (T& elem : res)
        elem = rand.nextGaussian(0, stdev);
    return res;
}


// ------------------------------ Tensor Views ------------------------------ //

// Returns a new view of the given tensor that is narrowed in
// the given dimension. The resulting tensor has the same shape
// as the original, with the exception that shape(dimension) == length.
//
// Ex: A
// 1  2  3  4
// 5  6  7  8
// 9 10 11 12
//
// narrow(A, 1, 1, 2)
//  2  3
//  6  7
// 10 11
template <class T>
Tensor<T> narrow(const Tensor<T>& tensor, const size_t dimension,
    const size_t startIndex, const size_t length)
{
    ASSERT(tensor.shape(dimension) >= startIndex + length,
        "Invalid narrowing range. start + length < shape(dimension).");

    // Calculate the new shape. It should be exactly
    // the same as ours with the exception that dimension
    // 'dimension' should have size 'length' instead.
    SmallVector shape(tensor.shape());
    shape[dimension] = length;

    // The stride is unchanged
    SmallVector stride(tensor.stride());

    // Calculate the new offset
    size_t offset = stride[dimension] * startIndex;

    return Tensor<T>(tensor.storage(), shape.begin(), shape.end(),
        stride.begin(), stride.end(), offset);
}


// Returns a new view of the given tensor that is effectively narrowed
// in multiple dimensions. Each element of 'beginEndPairs' is a pair
// of <start, end> indices for that dimension. If fewer than rank()
// pairs are given, only the first K dimensions will be affected.
// All other dimensions will be left alone.
//
// Ex: A
// 1  2  3  4
// 5  6  7  8
// 9 10 11 12
//
// sub(A, { {1, 2}, {2, 3} })
//  7  8
// 11 12
template <class T>
Tensor<T> sub(const Tensor<T>& tensor, initializer_list< pair<size_t, size_t> > beginEndPairs)
{
    ASSERT(beginEndPairs.size() <= tensor.rank(), "No more than rank() pairs may be provided.");

    SmallVector shape(tensor.shape());
    SmallVector stride(tensor.stride());
    size_t offset = 0;

    // Calculate the new shape and offset
    size_t index = 0;
    for (auto& elem : beginEndPairs)
    {
        shape[index] = elem.second - elem.first + 1;
        offset      += stride[index] * elem.first;
        ++index;
    }

    return Tensor<T>(tensor.storage(), shape.begin(), shape.end(),
        stride.begin(), stride.end(), offset);
}


// Returns a new view of the given tensor with exactly one fewer dimension.
//
// Ex: A
// 1  2  3  4
// 5  6  7  8
// 9 10 11 12
//
// select(A, 0, 1)
// 5  6  7  8
template <class T>
Tensor<T> select(const Tensor<T>& tensor, const size_t dimension, const size_t index)
{
    ASSERT(tensor.rank() >= 1, "Cannot use select() on a scalar.");

    // Calculate the new shape, stride, and offset
    SmallVector shape(tensor.shape());
    shape.erase(shape.begin() + dimension);

    SmallVector stride(tensor.stride());
    stride.erase(stride.begin() + dimension);

    size_t offset = tensor.stride(dimension) * index;

    return Tensor<T>(tensor.storage(), shape.begin(), shape.end(),
        stride.begin(), stride.end(), offset);
}

// Returns a new view of the given tensor with the given two dimensions switched.
// Both the shape and the stride are reversed, but the offset is not affected.
//
// Ex: A
// 1  2  3  4
// 5  6  7  8
// 9 10 11 12
//
// transpose(A, 0, 1)
// 1 5  9
// 2 6 10
// 3 7 11
// 4 8 12
template <class T>
Tensor<T> transpose(const Tensor<T>& tensor, const size_t dim1, const size_t dim2)
{
    ASSERT(dim1 < tensor.rank(), "Dim1 is invalid.");
    ASSERT(dim2 < tensor.rank(), "Dim2 is invalid.");

    SmallVector shape(tensor.shape());
    SmallVector stride(tensor.stride());

    using namespace std;
    swap(shape[dim1], shape[dim2]);
    swap(stride[dim1], stride[dim2]);

    return Tensor<T>(tensor.storage(), shape.begin(), shape.end(),
        stride.begin(), stride.end());
}


// Returns a new view of the given tensor where the dimensions have been
// permuted according to the given indices. This is a shorthand for
// calling transpose() repeatedly. All dimensions must be provided.
//
// Ex: A
// 1  2  3  4
// 5  6  7  8
// 9 10 11 12
//
// permute(A, {0, 1})
// 1 5  9
// 2 6 10
// 3 7 11
// 4 8 12
template <class T, class InputIt>
Tensor<T> permute(const Tensor<T>& tensor, InputIt begin, InputIt end)
{
    ASSERT(std::distance(begin, end) == (int) tensor.rank(),
        "All indices must be provided.");

    SmallVector shape(tensor.rank());
    SmallVector stride(tensor.rank());

    size_t index = 0;
    for (auto it = begin; it != end; ++it)
    {
        shape[index]  = tensor.shape(index);
        stride[index] = tensor.stride(index);
        index++;
    }

    return Tensor<T>(tensor.storage(), shape.begin(), shape.end(),
        stride.begin(), stride.end());
}

template <class T>
Tensor<T> permute(const Tensor<T>& tensor, initializer_list<size_t> indices)
{
    return permute(tensor, indices.begin(), indices.end());
}

// Returns a new view of the given tensor where singleton dimensions (either
// implicitly or explicitly defined) are effectively duplicated by setting
// the corresponding strides to 0.
//
// Ex: A, shape = [1, 4], stride = [4, 1]
// [ 1 2 3 4 ]
//
// expand(A, {3, 4} ), shape = [3, 4], stride = [0, 1]
// [ 1 2 3 4 ]
// [ 1 2 3 4 ]
// [ 1 2 3 4 ]
//
// Ex: B, shape = [], stride = []
// [5]
//
// expand(B, {2, 2} ), shape = [2, 2], stride = [0, 0]
// [ 5 5 ]
// [ 5 5 ]
template <class T, class InputIt>
Tensor<T> expand(const Tensor<T>& tensor, InputIt begin, InputIt end)
{
    const size_t newRank = std::distance(begin, end);

    ASSERT(newRank >= tensor.rank(),
        "expand() can only make a tensor larger, not smaller.");

    // Set any stride elements to 0 for which the size changed
    SmallVector stride(tensor.stride());

    auto newShapeIt = begin + (newRank - tensor.rank());
    auto strideIt   = stride.begin();
    for (const size_t& orig : tensor.shape())
    {
        if (orig != *newShapeIt)
        {
            ASSERT(orig == 1,
                "expand() can only be used for dimensions with a size of 1.");
            *strideIt = 0;
        }

        ++newShapeIt;
        ++strideIt;
    }

    // Insert 0 strides for any missing leading dimensions
    stride.insert(stride.begin(), newRank - tensor.rank(), 0);

    return Tensor<T>(tensor.storage(), begin, end,
        stride.begin(), stride.end());
}

template <class T>
Tensor<T> expand(const Tensor<T>& tensor, initializer_list<size_t> shape)
{
    return expand(tensor, shape.begin(), shape.end());
}

// Many times, an expansion might be necessary under some circumstances, and it
// might be unnecessary for others. For example, 'x * expand(1, shape)' can be
// simplified to simply 'x' when 'x.shape() > shape'. If 'x.shape() > shape',
// however, the expansion is necessary to ensure the result has the correct
// shape. This function will only expand the input if it is determined to be
// necessary. Otherwise, the input will be returned unchanged.
template <class T, class InputIt>
Tensor<T> expandIfSmaller(const Tensor<T>& tensor, InputIt begin, InputIt end)
{
    size_t oldRank = tensor.rank();
    size_t newRank = (size_t)(std::distance(begin, end));

    if (oldRank > newRank)
        return tensor;
    else if (oldRank < newRank)
        return expand(tensor, begin, end);
    else
    {
        const SmallVector& shape = tensor.shape();
        auto it1 = shape.begin();
        auto it2 = begin;

        while (it2 != end)
        {
            if (*it1 == 1 && *it2 != 1)
                return expand(tensor, begin, end);
            ++it1;
            ++it2;
        }
        return tensor;
    }
}

template <class T>
Tensor<T> expandIfSmaller(const Tensor<T>& tensor, initializer_list<size_t> shape)
{
    if (tensor.rank() <= shape.size())
        return expand(tensor, shape);
    else return tensor;
}

}
#endif

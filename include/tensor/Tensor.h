#ifndef TENSOR_H
#define TENSOR_H

#include <algorithm>
#include <functional>
#include <vector>
#include <initializer_list>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include "util/Assert.h"
#include "util/BufferAllocator.h"
#include "util/HeapAllocator.h"
#include "tensor/Storage.h"

namespace opkit
{

using std::vector;
using std::initializer_list;
using std::pair;

using SmallVector = std::vector<size_t, BufferAllocator<size_t, 2048>>;
// using SmallVector = std::vector<size_t, HeapAllocator<size_t, 1024>>;
// using SmallVector = std::vector<size_t>;

#include "tensor/TensorIterator.h"

template <class T>
class Tensor
{
public:
    typedef TensorIterator<T, false> iterator;
    typedef TensorIterator<T, true> const_iterator;

    // Default constructor. Creates a scalar Tensor
    // with a new storage.
    Tensor();

    // Create a new Tensor with the given shape/stride.
    // New storage will be allocated. Initializer lists allow
    // for syntax like this: "Tensor<double>({3, 5})", while
    // iterators allow for other containers to be used instead.
    Tensor(std::initializer_list<size_t> shape);
    Tensor(std::initializer_list<size_t> shape,
        std::initializer_list<size_t> stride);

    template <class InputIt>
    Tensor(InputIt shapeBegin, InputIt shapeEnd);
    template <class ShapeIt, class StrideIt>
    Tensor(ShapeIt shapeBegin, ShapeIt shapeEnd,
        StrideIt strideBegin, StrideIt strideEnd);

    // Create a new Tensor using the given storage. The storage
    // offset, shape, and stride can all be provided or ignored.
    // When only the storage is provided, a 1D tensor is created.
    //
    // As with the other constructors, either initializer lists
    // or iterator pairs may be used to provide values.
    Tensor(const Storage<T>& storage, const size_t offset = 0);
    Tensor(const Storage<T>& storage, std::initializer_list<size_t> shape,
        const size_t offset = 0);
    Tensor(const Storage<T>& storage,
        std::initializer_list<size_t> shape,
        std::initializer_list<size_t> stride,
        const size_t offset = 0);

    template <class InputIt>
    Tensor(const Storage<T>& storage, InputIt shapeBegin,
        InputIt shapeEnd, const size_t offset = 0);
    template <class ShapeIt, class StrideIt>
    Tensor(const Storage<T>& storage,
        ShapeIt shapeBegin, ShapeIt shapeEnd,
        StrideIt strideBegin, StrideIt strideEnd,
        const size_t offset = 0);

    Tensor(const Tensor<T>& orig)                = default;
    Tensor(Tensor<T>&& orig)                     = default;
    Tensor<T>& operator=(const Tensor<T>& other) = default;
    Tensor<T>& operator=(Tensor<T>&& other)      = default;

    // Create a new scalar tensor with the given value.
    template <class U>
    static Tensor<T> fromScalar(const U& scalar)
    {
        Tensor<T> res;
        res = T(scalar);
        return res;
    }

    // Create a new rank-1 tensor (vector) with the given values. The values do
    // not have to be the same type as the tensor, but they will be converted.
    template <class U>
    static Tensor<T> fromVector(std::initializer_list<U> data)
    {
        return Tensor<T>(Storage<T>(data));
    }

    template <class U, class Alloc>
    static Tensor<T> fromVector(const std::vector<U, Alloc>& data)
    {
        return Tensor<T>(Storage<T>(data.begin(), data.end()));
    }

    // Create a new Tensor of arbitrary shape with the given values.
    template <class U>
    static Tensor<T> fromValues(std::initializer_list<U> data, std::initializer_list<size_t> shape)
    {
        return Tensor<T>(Storage<T>(data), shape);
    }

    template <class U, class Alloc>
    static Tensor<T> fromValues(const std::vector<U, Alloc>& data, std::initializer_list<size_t> shape)
    {
        return Tensor<T>(Storage<T>(data.begin(), data.end()), shape);
    }

    // The () operator provides a simple mechanism for isolating
    // the K leading dimensions of a tensor by returning a new view
    // with fewer dimensions.
    //
    // Example: If T is a <2, 2, 3> tensor containing the following,
    // [ 0  1  2 ]
    // [ 6  7  8 ]
    //
    // [ 12 13 14 ]
    // [ 18 19 20 ]
    //
    // T({0})       = [ 0 1 2 ] [ 6 7 8 ]
    // T({1})       = [ 12 13 14 ] [ 18 19 20 ]
    // T({0, 0})    = [ 0 1 2 ]
    // T({0, 1})    = [ 6 7 8 ]
    // T({0, 0, 0}) = [ 0 ]
    // T({1, 1, 2}) = [ 20 ]
    //
    // Note: If the number of indices == rank(), a rank-0 tensor
    // is returned.
    Tensor<T> operator()(initializer_list<size_t> indices) const;

    template <class InputIt>
    Tensor<T> operator()(InputIt indicesBegin, InputIt indicesEnd) const;

    // ONLY VALID FOR RANK-0 TENSORS!
    //
    // Rank-0 tensors are effectively scalars. These functions allow
    // normal scalar-like operations to be used with tensors. For
    // example: T({3, 4}) = 5.
    //
    // NOTE: An assertion will guarantee that only rank-0 tensors
    // are used. The assertion will fail if the rank is > 0.
    explicit operator T() const;
    Tensor<T>& operator=(const T& value);

    // Access a particular cell of the tensor. All indices
    // must be provided.
    T& at(initializer_list<size_t> indices);
    const T& at(initializer_list<size_t> indices) const;

    template <class InputIt>
    T& at(InputIt indicesBegin, InputIt indicesEnd);

    template <class InputIt>
    const T& at(InputIt indicesBegin, InputIt indicesEnd) const;

    // Iterators
    iterator begin();
    const_iterator begin() const;
    iterator end();
    const_iterator end() const;

    // Create a new Tensor containing the same information as
    // this object. New memory will be allocated. The template
    // parameter U controls the destination type. If the
    // destination type does not match the original type, the
    // appropriate type conversion will be used on each element
    // of the original Tensor object (e.g. casting from
    // Tensor<double> to Tensor<int>). Note that it is
    // possible for a type conversion to lose information.
    template <class U = T>
    Tensor<U> clone() const;

    // Copy the contents of this tensor into 'other', possibly resizing
    // it if necessary.
    //
    // NOTE: It is the user's responsibility to ensure that 'other' does not
    // share its data with another Tensor. If it does, it is possible for the
    // other Tensor to be modified inadvertently.
    template <class U = T>
    void copy(Tensor<U>& other) const;

    // Copy the given value into every cell of the Tensor.
    // Returns *this so operations may be chained if desired.
    Tensor<T>& fill(const T& value);

    // Applies the given function (anything with a () operator
    // that takes a T and returns a T) to this tensor. Returns
    // *this so operations may be chained if desired.
    template <class Function>
    Tensor<T>& apply(Function&& f);

    // Returns true when this tensor is fully contiguous
    bool contiguous() const;

    // Resize this tensor, including the underlying storage. The number of
    // elements and the strides will be calculated automatically, but the
    // desired shape must be provided. Note that calling this function will
    // break any connections to other tensors or Storage objects.
    template <class InputIt>
    void resize(InputIt shapeBegin, InputIt shapeEnd);
    void resize(std::initializer_list<size_t> shape);

    // Data access
    Storage<T>&       storage()       { return mStorage;        }
    const Storage<T>& storage() const { return mStorage;        }
    T*                data()          { return mStorage.data(); }
    const T*          data()    const { return mStorage.data(); }

    // Getters
    size_t             shape(const size_t dim)  const { return mShape[dim];    }
    const SmallVector& shape()                  const { return mShape;         }
    size_t             rank()                   const { return mShape.size();  }
    size_t             stride(const size_t dim) const { return mStride[dim];   }
    const SmallVector& stride()                 const { return mStride;        }
    size_t             size()                   const { return mNumElements;   }

private:
    SmallVector mShape;
    SmallVector mStride;
    size_t mNumElements;
    Storage<T> mStorage;
};


//-------------------------------------------------------------//
// Helpers
//-------------------------------------------------------------//

SmallVector calculateOptimalStride(const SmallVector& shape)
{
    SmallVector res(shape.size());
    size_t multiplier = 1;
    size_t index      = shape.size() - 1;

    for (auto shapeIt = shape.rbegin(); shapeIt != shape.rend(); ++shapeIt)
    {
        res[index--] = multiplier;
        multiplier  *= (*shapeIt);
    }
    return res;
}

void calculateOptimalStride(SmallVector& res, const SmallVector& shape)
{
    res.resize(shape.size());
    size_t multiplier = 1;
    size_t index      = shape.size() - 1;

    for (auto shapeIt = shape.rbegin(); shapeIt != shape.rend(); ++shapeIt)
    {
        res[index--] = multiplier;
        multiplier  *= (*shapeIt);
    }
}

size_t countElements(const SmallVector& shape)
{
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
}

//-------------------------------------------------------------//
// Constructors
//-------------------------------------------------------------//

template <class T>
Tensor<T>::Tensor() :
    Tensor<T>({1})
{}

template <class T>
Tensor<T>::Tensor(std::initializer_list<size_t> shape) :
    Tensor<T>(shape.begin(), shape.end())
{}

template <class T>
Tensor<T>::Tensor(std::initializer_list<size_t> shape,
    std::initializer_list<size_t> stride) :
    Tensor<T>(shape.begin(), shape.end(), stride.begin(), stride.end())
{}

template <class T>
template <class InputIt>
Tensor<T>::Tensor(InputIt shapeBegin, InputIt shapeEnd) :
    mShape(shapeBegin, shapeEnd),
    mStride(calculateOptimalStride(mShape)),
    mNumElements(countElements(mShape)),
    mStorage(mNumElements)
{
    ASSERT(std::distance(shapeBegin, shapeEnd) > 0, "Shape cannot be empty.");
}

template <class T>
template <class ShapeIt, class StrideIt>
Tensor<T>::Tensor(ShapeIt shapeBegin, ShapeIt shapeEnd,
    StrideIt strideBegin, StrideIt strideEnd) :
    mShape(shapeBegin, shapeEnd),
    mStride(strideBegin, strideEnd),
    mNumElements(countElements(mShape)),
    mStorage(mNumElements)
{
    ASSERT(std::distance(shapeBegin, shapeEnd) > 0, "Shape cannot be empty.");
    ASSERT(std::distance(shapeBegin, shapeEnd) == std::distance(strideBegin, strideEnd),
        "Shape and stride must have the same # of dimensions.");
}

template <class T>
Tensor<T>::Tensor(const Storage<T>& storage,
    const size_t offset) :
    mShape({storage.size()}),
    mStride({1}),
    mNumElements(storage.size()),
    mStorage(storage, offset)
{}

template <class T>
Tensor<T>::Tensor(const Storage<T>& storage,
    std::initializer_list<size_t> shape,
    const size_t offset) :
    Tensor<T>(storage, shape.begin(), shape.end(), offset)
{}

template <class T>
Tensor<T>::Tensor(const Storage<T>& storage,
    std::initializer_list<size_t> shape,
    std::initializer_list<size_t> stride,
    const size_t offset) :
    Tensor<T>(storage, shape.begin(), shape.end(),
        stride.begin(), stride.end(), offset)
{}

template <class T>
template <class InputIt>
Tensor<T>::Tensor(const Storage<T>& storage, InputIt shapeBegin,
    InputIt shapeEnd, const size_t offset) :
    mShape(shapeBegin, shapeEnd),
    mStride(calculateOptimalStride(mShape)),
    mNumElements(countElements(mShape)),
    mStorage(storage, offset)
{
    ASSERT(std::distance(shapeBegin, shapeEnd) > 0, "Shape cannot be empty.");
}

template <class T>
template <class ShapeIt, class StrideIt>
Tensor<T>::Tensor(const Storage<T>& storage,
    ShapeIt shapeBegin, ShapeIt shapeEnd,
    StrideIt strideBegin, StrideIt strideEnd,
    const size_t offset) :
    mShape(shapeBegin, shapeEnd),
    mStride(strideBegin, strideEnd),
    mNumElements(countElements(mShape)),
    mStorage(storage, offset)
{
    ASSERT(std::distance(shapeBegin, shapeEnd) > 0, "Shape cannot be empty.");
    ASSERT(std::distance(shapeBegin, shapeEnd) == std::distance(strideBegin, strideEnd),
        "Shape and stride must have the same # of dimensions.");
}



//-------------------------------------------------------------//
// Tensor Views
//-------------------------------------------------------------//

template <class T>
Tensor<T> Tensor<T>::operator()(initializer_list<size_t> indices) const
{
    return (*this)(indices.begin(), indices.end());
}

template <class T>
template <class InputIt>
Tensor<T> Tensor<T>::operator()(InputIt indicesBegin, InputIt indicesEnd) const
{
    int size = std::distance(indicesBegin, indicesEnd);

    // Remove the first 'size' components of the shape
    SmallVector shape(mShape);
    shape.erase(shape.begin(), shape.begin() + size);

    // Remove the first 'size' components of the stride
    SmallVector stride(mStride);
    stride.erase(stride.begin(), stride.begin() + size);

    // Calculate the new offset
    size_t offset = 0;
    auto strideIt = mStride.begin();

    for (auto& indicesIt = indicesBegin; indicesIt != indicesEnd; ++indicesIt)
        offset += (*indicesIt) * (*strideIt++);

    // Create a new view with the given parameters
    return Tensor<T>(mStorage, shape.begin(), shape.end(),
        stride.begin(), stride.end(), offset);
}

//-------------------------------------------------------------//
// Element Access
//-------------------------------------------------------------//

template <class T>
Tensor<T>::operator T() const
{
    ASSERT(size() == 1, "Only constants can be read with the () operator.");
    return mStorage[0];
}

template <class T>
Tensor<T>& Tensor<T>::operator=(const T& value)
{
    ASSERT(size() == 1, "Only constants can be assigned directly.");
    mStorage[0] = value;
    return *this;
}

template <class T>
T& Tensor<T>::at(initializer_list<size_t> indices)
{
    return at(indices.begin(), indices.end());
}

template <class T>
const T& Tensor<T>::at(initializer_list<size_t> indices) const
{
    return at(indices.begin(), indices.end());
}

template <class T>
template <class InputIt>
T& Tensor<T>::at(InputIt indicesBegin, InputIt indicesEnd)
{
    return const_cast<T&>
    (
        static_cast<const Tensor<T>*>(this)->at(indicesBegin, indicesEnd)
    );
}

template <class T>
template <class InputIt>
const T& Tensor<T>::at(InputIt indicesBegin, InputIt indicesEnd) const
{
    // Guarantee the ranks match
    ASSERT(std::distance(indicesBegin, indicesEnd) == (int)rank(),
        "at(): Number of indices given does not match tensor shape.");

    size_t index  = 0;
    auto strideIt = mStride.begin();
    for (auto indicesIt = indicesBegin; indicesIt != indicesEnd; ++indicesIt)
        index += (*indicesIt) * (*strideIt++);

    return mStorage[index];
}

//-------------------------------------------------------------//
// Iterators
//-------------------------------------------------------------//

template <class T>
typename Tensor<T>::iterator Tensor<T>::begin()
{
    return Tensor<T>::iterator(this);
}

template <class T>
typename Tensor<T>::const_iterator Tensor<T>::begin() const
{
    return Tensor<T>::const_iterator(this);
}

template <class T>
typename Tensor<T>::iterator Tensor<T>::end()
{
    return Tensor<T>::iterator(this, true);
}

template <class T>
typename Tensor<T>::const_iterator Tensor<T>::end() const
{
    return Tensor<T>::const_iterator(this, true);
}

//-------------------------------------------------------------//
// Additional Methods
//-------------------------------------------------------------//

template <class T>
template <class U>
Tensor<U> Tensor<T>::clone() const
{
    // Create new contiguous storage for the clone
    Storage<U> storage(mNumElements);

    // Fill the storage with the elements from this tensor
    if (contiguous())
    {
        const T* start = mStorage.begin();
        std::copy(start, start + mNumElements, storage.begin());
    }
    else
    {
        size_t i = 0;
        for (const T& elem : *this)
            storage[i++] = U(elem);
    }

    // Use the source's shape, but let the stride be inferred
    return Tensor<U>(storage, mShape.begin(), mShape.end());
}

template <class T>
template <class U>
void Tensor<T>::copy(Tensor<U>& other) const
{
    // 1. Resize 'other''s storage (if necessary)
    if (other.mStorage.size() < mNumElements)
        other.mStorage.resize(mNumElements);

    // 2. Copy the data and metadata to 'other'
    if (contiguous())
    {
        const T* start = mStorage.begin();
        std::copy(start, start + mNumElements, other.mStorage.begin());

        other.mShape.assign(mShape.begin(), mShape.end());
        other.mStride.assign(mStride.begin(), mStride.end());
        other.mNumElements = mNumElements;
    }
    else if (rank() == 2)
    {
        const T* src  = mStorage.begin();
              T* dest = other.mStorage.begin();

        const size_t width   = shape(1);
        const size_t height  = shape(0);
        const size_t xStride = stride(1);
        const size_t yStride = stride(0);

        for (size_t y = 0; y < height; ++y)
        {
            const T* row = src;
            for (size_t x = 0; x < width; ++x)
            {
                *dest++ = U(*row);
                row += xStride;
            }
            src += yStride;
        }

        other.mShape.assign(mShape.begin(), mShape.end());
        calculateOptimalStride(other.mStride, other.mShape);
        other.mNumElements = mNumElements;
    }
    else
    {
        size_t i = 0;
        for (const T& elem : *this)
            other.mStorage[i++] = U(elem);

        other.mShape.assign(mShape.begin(), mShape.end());
        calculateOptimalStride(other.mStride, other.mShape);
        other.mNumElements = mNumElements;
    }
}

template <class T>
Tensor<T>& Tensor<T>::fill(const T& value)
{
    // Avoid iterators whenever possible to improve performance
    if (contiguous())
    {
        T* start = mStorage.begin();
        std::fill(start, start + mNumElements, value);
    }
    else
    {
        for (T& elem : *this)
            elem = value;
    }

    return *this;
}

template <class T>
template <class Function>
Tensor<T>& Tensor<T>::apply(Function&& f)
{
    // Avoid iterators whenever possible to improve performance

    // Contiguous array optimization
    if (contiguous())
    {
        T* it  = mStorage.begin();
        T* end = it + mNumElements;
        while (it != end)
        {
            T& elem = *it;
            elem    = f(elem);
            ++it;
        }
    }

    // Vector optimization
    else if (rank() == 1)
    {
        const size_t stride = mStride[0];
        T* it               = mStorage.begin();
        for (size_t i = 0; i < mNumElements; ++i)
        {
            *it = f(*it);
            it += stride;
        }
    }

    // Matrix optimization
    else if (rank() == 2)
    {
        const size_t xStride = mStride[1];
        const size_t yStride = mStride[0];
        T* it                = mStorage.begin();
        for (size_t y = 0; y < mShape[0]; ++y)
        {
            T* row = it;
            for (size_t x = 0; x < mShape[1]; ++x)
            {
                *row = f(*row);
                row += xStride;
            }
            it += yStride;
        }
    }

    // Generic implementation
    else
    {
        for (T& elem : *this)
            elem = f(elem);
    }

    return *this;
}

template <class T>
bool Tensor<T>::contiguous() const
{
    size_t stride = 1;
    for (int i = rank(); i > 0; --i)
    {
        if (mStride[i - 1] != stride) return false;
        stride *= mShape[i - 1];
    }
    return true;
}

template <class T>
template <class InputIt>
void Tensor<T>::resize(InputIt shapeBegin, InputIt shapeEnd)
{
    // Only resize if the shape has actually changed
    size_t index = 0;
    for (InputIt it = shapeBegin; it != shapeEnd; ++it)
    {
        if (*it != mShape[index++])
        {
            mShape.assign(shapeBegin, shapeEnd);
            mStride      = calculateOptimalStride(mShape);
            mNumElements = countElements(mShape);
            mStorage.resize(mNumElements);
            break;
        }
    }
}

template <class T>
void Tensor<T>::resize(std::initializer_list<size_t> shape)
{
    resize(shape.begin(), shape.end());
}

// Allow ranges to be printed
template <class InputIt>
std::string to_string(InputIt begin, InputIt end)
{
    std::string res("[ ");
    for (auto it = begin; it != end; ++it)
        res.append(std::to_string(*it) + ", ");

    res.pop_back();
    res.pop_back();
    res.append(" ]");
    return res;
}

// Allow vectors to be printed. This is used for assertions and for printing
// properties of tensors.
template <class T, class Alloc>
std::string to_string(const std::vector<T, Alloc>& vec)
{
    return to_string(vec.begin(), vec.end());
}

template <class T, class Alloc>
std::ostream& operator<<(std::ostream& out, const vector<T, Alloc>& vec)
{
    out << to_string(vec);
    return out;
}



// // Common implementation for both versions of reshape() above
// template <class InputIt>
// void reshape(InputIt begin, InputIt end)
// {
    // // Calculate size of new shape
    // size_t numElements = 1;
    // for (auto it = begin; it != end; ++it)
        // numElements *= (*it);

    // if (numElements != 0)
    // {
        // // Ensure everything is OK
        // assert(("Number of elements cannot be changed with a reshape.",
            // numElements == mNumElements));

        // mShape.clear();
        // mShape.insert(mShape.begin(), begin, end);
    // }

    // // Deal with missing values
    // else
    // {
        // vector<size_t> values(begin, end);
        // size_t partialProduct = 1;
        // int count             = 0;
        // size_t zeroIndex      = 0;
        // for (size_t i = 0; i < values.size(); ++i)
        // {
            // if (values[i] != 0)
                // partialProduct *= values[i];
            // else
            // {
                // ++count;
                // zeroIndex = i;
            // }
        // }

        // // We can't deal with 2 or more 0 elements
        // assert(("At most 1 dimension can be inferred.", count <= 1));

        // // It has to be possible to fill in the missing value
        // assert(("Not possible to infer dimension.",
            // mNumElements % partialProduct == 0));

        // values[zeroIndex] = mNumElements / partialProduct;
        // mShape.clear();
        // mShape.insert(mShape.begin(), values.begin(), values.end());
    // }
// }

// // Common implementation for both versions of resize() above
// template <class InputIt>
// void resize(InputIt begin, InputIt end)
// {
    // mNumElements = 1;
    // for (auto it = begin; it != end; ++it)
        // mNumElements *= (*it);

    // // We do not allow partial declaration of the shape
    // assert(("Partial declaration not allowed in resize().",
        // mNumElements != 0));

    // mShape.clear();
    // mShape.insert(mShape.begin(), begin, end);

    // // Calculate the stride
    // mStride.resize(mShape.size());
    // size_t multiplier = 1;
    // size_t index      = mStride.size() - 1;
    // for (auto shapeIt = mShape.end() - 1; shapeIt != mShape.begin() - 1; --shapeIt)
    // {
        // mStride[index--] = multiplier;
        // multiplier *= (*shapeIt);
    // }

    // // Reset the offset
    // mOffset = 0;
// }

}

#endif

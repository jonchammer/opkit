#ifndef STORAGE_H
#define STORAGE_H

#include <initializer_list>
#include "util/ReferenceCount.h"

namespace tensorlib
{

// This class represents a 1D array of type T. Unlike a normal
// array, instances of this class employ reference counting
// to share data and to perform automatic garbage collection.
// The non-copy constructor(s), clone() method, and resize()
// method are the only functions that will allocate any new
// space. All other functions will either operate on the object
// in place or will return a view that internally references
// the same location in memory. The model is very similar to
// that used in Torch.
template <class T>
class Storage
{
public:

    // Create a new Storage of size 0. Memory will be allocated
    // but will not be initialized.
    Storage();

    // Create a new Storage of the given size. Memory will be
    // allocated but will not be initialized.
    explicit Storage(const size_t size);

    // Create a new Storage containing the given values. Memory
    // will be allocated and initialized. For the first version,
    // any class with a begin()/end() pair can be used.
    template <class InputIt>
    Storage(InputIt begin, InputIt end);

    template <class U>
    Storage(std::initializer_list<U> values);

    // Create a new view of the given storage. The same memory
    // will be used internally, so no copies will be made.
    // If provided, the offset allows the view to start at an
    // index other than 0. All offsets are relative with respect
    // to the parent view (or the original Storage object).
    // By default, the size of the view will be set such that
    // the view extends to the end of the parent, but it can
    // be set manually.
    Storage(const Storage<T>& orig, const size_t offset = 0);
    Storage(const Storage<T>& orig, const size_t offset, const size_t size);

    // Move construction
    Storage(Storage<T>&& orig) = default;

    // Assignment
    Storage<T>& operator=(const Storage<T>& orig) = default;
    Storage<T>& operator=(Storage<T>&& orig)      = default;

    // Access a single element of the Storage. All indices are
    // relative to the view (if applicable).
    T& operator[](const size_t index);
    const T& operator[](const size_t index) const;

    // Access the first and last elements of the collection
    T& front();
    const T& front() const;
    T& back();
    const T& back() const;

    // Standard iterators
    T* begin();
    const T* begin() const;
    T* end();
    const T* end() const;

    // Returns a pointer to the underlying buffer
    T* data();
    const T* data() const;

    // Create a new Storage containing the same information as
    // this object. New memory will be allocated. The template
    // parameter U controls the destination type. If the
    // destination type does not match the original type, the
    // appropriate type conversion will be used on each element
    // of the original Storage object (e.g. casting from
    // Storage<double> to Storage<int>). Note that it is
    // possible for a type conversion to lose information.
    template <class U = T>
    Storage<U> clone() const;

    // Copy the given value into a certain range of the Storage
    // (the entire Storage in the first version). Returns *this
    // so operations may be chained if desired.
    Storage<T>& fill(const T& value);
    Storage<T>& fill(const size_t start, const size_t end, const T& value);

    // Replace the underlying buffer used to hold the information
    // with a new buffer of the given size. The contents of the
    // new buffer are undetermined. The buffer is effectively
    // uninitialized. Returns *this so operations may be chained
    // if desired.
    //
    // NOTE: If 'size' matches the current size of the buffer, this
    // function does nothing. The same object is returned unmodified.
    //
    // NOTE: This function breaks any connection with other Storage
    // objects. If two Storage objects use the same underlying buffer
    // and one of them is resized, the other Storage object will
    // retain sole ownership of the original buffer.
    Storage<T>& resize(const size_t size);

    // Returns the size of this storage.
    size_t size() const { return mSize; }

    // Returns true if the storage is empty.
    bool empty() const { return mSize == 0; }

    // Returns the absolute offset of this storage.
    size_t offset() const { return mOffset; }

private:

    // The actual data buffer. Since it derives from
    // RCObject, it has reference counting semantics.
    struct Buffer : public RCObject
    {
        T* data;
        size_t size;

        Buffer(const size_t s) :
            data(new T[s]), size(s) {}

        template <class InputIt>
        Buffer(InputIt begin, InputIt end) :
            data(new T[std::distance(begin, end)]), size(std::distance(begin, end))
        {
            std::copy(begin, end, data);
        }

        template <class U>
        Buffer(std::initializer_list<U> values) :
            data(new T[values.size()]), size(values.size())
        {
            std::copy(values.begin(), values.end(), data);
        }

        Buffer(const Buffer& orig): RCObject(orig),
            data(new T[orig.size]), size(orig.size)
        {
            std::copy(orig.data, orig.data + size, data);
        }

        Buffer(Buffer&& orig) noexcept :
            data(orig.data), size(orig.size)
        {
            orig.data = nullptr;
            orig.size = 0;
        }

        ~Buffer() noexcept { if (data != nullptr) delete[] data; }

        Buffer& operator=(const Buffer& other)
        {
            // Reuse the copy constructor and move assignment operators
            Buffer tmp(other);
            *this = std::move(tmp);
            return *this;
        }

        Buffer& operator=(Buffer&& other) noexcept
        {
            if (data != nullptr) delete[] data;
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
            return *this;
        }

        Buffer* clone() const
        {
            return new Buffer(*this);
        }
    };

    RCPtr<Buffer> mValue; // Pointer to the buffer used.
    size_t mOffset;       // Offset of this particular object into the buffer.
    size_t mSize;         // Size of this particular object.
};

//--------------------------------------------------------//
// Constructors
template <class T>
Storage<T>::Storage() :
    mValue(new Buffer(0)), mOffset(0), mSize(0)
{}

template <class T>
Storage<T>::Storage(const size_t size) :
    mValue(new Buffer(size)), mOffset(0), mSize(size)
{}

template <class T>
template <class InputIt>
Storage<T>::Storage(InputIt begin, InputIt end) :
    mValue(new Buffer(begin, end)), mOffset(0), mSize(std::distance(begin, end))
{}

template <class T>
template <class U>
Storage<T>::Storage(std::initializer_list<U> values) :
    mValue(new Buffer(values)), mOffset(0), mSize(values.size())
{}

template <class T>
Storage<T>::Storage(const Storage<T>& orig, const size_t offset) :
    mValue(orig.mValue), mOffset(orig.mOffset + offset),
    mSize(orig.mOffset + orig.mSize - mOffset) // Make sure views are sized correctly
{}

template <class T>
Storage<T>::Storage(const Storage<T>& orig, const size_t offset, const size_t size) :
    mValue(orig.mValue), mOffset(orig.mOffset + offset), mSize(size) {}

// Element Access
template <class T>
T& Storage<T>::operator[](const size_t index)
{
    return mValue->data[mOffset + index];
}

template <class T>
const T& Storage<T>::operator[](const size_t index) const
{
    return mValue->data[mOffset + index];
}

template <class T>
T& Storage<T>::front()
{
    return mValue->data[mOffset];
}

template <class T>
const T& Storage<T>::front() const
{
    return mValue->data[mOffset];
}

template <class T>
T& Storage<T>::back()
{
    return mValue->data[mOffset + mSize - 1];
}

template <class T>
const T& Storage<T>::back() const
{
    return mValue->data[mOffset + mSize - 1];
}

// Iterators
template <class T>
T* Storage<T>::begin()
{
    return mValue->data + mOffset;
}

template <class T>
const T* Storage<T>::begin() const
{
    return mValue->data + mOffset;
}

template <class T>
T* Storage<T>::end()
{
    return mValue->data + mOffset + mSize;
}

template <class T>
const T* Storage<T>::end() const
{
    return mValue->data + mOffset + mSize;
}

template <class T>
T* Storage<T>::data()
{
    return mValue->data + mOffset;
}

template <class T>
const T* Storage<T>::data() const
{
    return mValue->data + mOffset;
}

// Additional methods
template <class T>
template <class U>
Storage<U> Storage<T>::clone() const
{
    Storage<U> res(mSize);
    for (size_t i = 0; i < mSize; ++i)
        res[i] = U(mValue->data[mOffset + i]);
    return res;
}

template <class T>
Storage<T>& Storage<T>::fill(const T& value)
{
    fill(0, mSize - 1, value);
    return *this;
}

template <class T>
Storage<T>& Storage<T>::fill(const size_t start, const size_t end, const T& value)
{
    for (size_t i = start; i <= end; ++i)
        mValue->data[mOffset + i] = value;
    return *this;
}

template <class T>
Storage<T>& Storage<T>::resize(const size_t size)
{
    if (size != mSize)
    {
        mValue  = new Buffer(size);
        mSize   = size;
        mOffset = 0;
    }

    return *this;
}

}

#endif

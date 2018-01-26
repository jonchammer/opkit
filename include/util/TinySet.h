#ifndef TINY_SET_H
#define TINY_SET_H

#include <functional>

namespace opkit
{

//===================================================//
// The Tiny Set Class Definition                     //
//===================================================//
template <class T>
class TinySet
{
public:
    // Constructors / Destructors
    TinySet();
    TinySet(const int capacity);
    TinySet(const TinySet<T>& orig);
    ~TinySet();

    // Modification
    void insert(const T& value);
    bool search(const T& value);
    bool remove(const T& value);
    void clear();

    // Status
    bool isFull()  const;
    bool isEmpty() const;
    int getSize()  const;

private:
    static const int DEFAULT_SIZE = 32;
    static const char FULL        = 'F';
    static const char EMPTY       = 'E';
    static const char AVAILABLE   = 'A';

    T* mData;
    char* mState;
    int mCapacity;
    int mSize;

    int indexOf(const T& value) const;
    void rehash();
};

//===================================================//
// Tiny Set Class Implementations                    //
//===================================================//

template <class T>
TinySet<T>::TinySet() :
    mData(new T[DEFAULT_SIZE]),
    mState(new char[DEFAULT_SIZE]),
    mCapacity(DEFAULT_SIZE),
    mSize(0)
{
    std::fill(mState, mState + mCapacity, EMPTY);
}

template <class T>
TinySet<T>::TinySet(const int capacity) :
    mData(new T[capacity]),
    mState(new char[capacity]),
    mCapacity(capacity),
    mSize(0)
{
    std::fill(mState, mState + mCapacity, EMPTY);
}

template <class T>
TinySet<T>::TinySet(const TinySet<T>& orig) :
    mData(new T[orig.mCapacity]),
    mState(new char[orig.mCapacity]),
    mCapacity(orig.mCapacity),
    mSize(orig.mSize)
{
    std::copy(orig.mData,  orig.mData  + mCapacity, mData);
    std::copy(orig.mState, orig.mState + mCapacity, mState);
}

template <class T>
TinySet<T>::~TinySet()
{
    delete[] mData;
    delete[] mState;
    mData  = NULL;
    mState = NULL;
}

//===================================================//

// This is a helper function for the search() and remove() functions.
// It will search the table for a cell with the given value and return
// the index at which that cell was found. If the given value is not
// present, -1 will be returned.
template <class T>
int TinySet<T>::indexOf(const T& value) const
{
    // Determine where to start looking
    int index = std::hash<T>{}(value) % mCapacity;
    for (int i = 0; i < mCapacity; ++i)
    {
        // If we found what we were looking for, return the index
        if (mState[index] == FULL && mData[index] == value)
            return index;

        // If we find an empty cell, we can simply stop
        else if (mState[index] == EMPTY)
            return -1;

        // Apply collision resolution scheme
        else index = (index + 1) % mCapacity;
    }

    // We wrapped back around to the original index, so we can stop.
    return -1;
}

// Doubles the capacity of the table and moves all elements
template <class T>
void TinySet<T>::rehash()
{
    // Allocate the new data
    size_t newCapacity = mCapacity * 2;
    T* newData         = new T[newCapacity];
    char* newState     = new char[newCapacity];
    std::fill(newState, newState + newCapacity, EMPTY);
    std::hash<T> hasher{};

    // Perform the rehash
    for (size_t i = 0; i < mCapacity; ++i)
    {
        if (mState[i] == FULL)
        {
            int newIndex = hasher(mData[i]) % newCapacity;
            while (newState[newIndex] == FULL)
                newIndex = (newIndex + 1) % newCapacity;
            newState[newIndex] = FULL;
            newData[newIndex]  = mData[i];
        }
    }

    // Update all the internal members
    delete[] mData;
    delete[] mState;
    mData     = newData;
    mState    = newState;
    mCapacity = newCapacity;
}

//===================================================//

template <class T>
void TinySet<T>::insert(const T& value)
{
    // We can't insert if the table is full
    if (isFull()) rehash();

    // Determine the first available cell
    int index = std::hash<T>{}(value) % mCapacity;
    while (mState[index] == FULL && mData[index] != value)
        index = (index + 1) % mCapacity;

    // Perform the insertion
    if (mState[index] != FULL)
    {
        mState[index] = FULL;
        mData[index]  = value;
        mSize++;
    }
}

template <class T>
bool TinySet<T>::search(const T& value)
{
    return indexOf(value) != -1;
}

template <class T>
bool TinySet<T>::remove(const T& value)
{
    int index = indexOf(value);
    if (index != -1)
    {
        // Cells that have been removed are AVAILABLE. They can be reused
        // in another insertion, but they cannot be used as stopping
        // criteria for searches.
        mState[index] = AVAILABLE;
        mSize--;
        return true;
    }
    else return false;
}

template <class T>
void TinySet<T>::clear()
{
    std::fill(mState, mState + mCapacity, EMPTY);
    mSize = 0;
}

//===================================================//

template <class T>
bool TinySet<T>::isFull() const
{
    return mSize >= mCapacity;
}

template <class T>
bool TinySet<T>::isEmpty() const
{
    return mSize <= 0;
}

template <class T>
int TinySet<T>::getSize() const
{
    return mSize;
}

}
#endif

#ifndef RANDOM_INDEX_ITERATOR_H
#define RANDOM_INDEX_ITERATOR_H

#include <algorithm>
#include "Rand.h"

namespace opkit
{

// This class provides a simple interface for iterating over a set of indices
// (values between 0 and N - 1) in a random order.
class RandomIndexIterator
{
public:

    // Create a new RandomIndexIterator. 'length' is the total number of indices
    // and usually represents the size of the collection whose values we wish
    // to examine in a random order.
    RandomIndexIterator(const size_t length) :
        mIndices(new size_t[length]), mLength(length), mCur(0)
    {
        for (size_t i = 0; i < length; ++i)
            mIndices[i] = i;
    }

    // Destructor
    ~RandomIndexIterator()
    {
        delete[] mIndices;
        mIndices = nullptr;
    }

    // Shuffles the indices using the given RNG.
    void reset(Rand& rand)
    {
        std::shuffle(mIndices, mIndices + mLength, rand.getGenerator());
        mCur = 0;
    }

    // Returns true if there are more indices to examine. When this function
    // returns false, the iterator will need to be reset before it can be used
    // again.
    bool hasNext()
    {
        return mCur < mLength;
    }

    // Returns the next random index. Make sure to call hasNext() first to
    // ensure there is a next value first.
    size_t next()
    {
        return mIndices[mCur++];
    }

    // Returns the current cursor position.
    size_t getCurrentPosition() const
    {
        return mCur;
    }

    // Returns the number of remaining indices.
    size_t getRemaining() const
    {
        return mLength - mCur;
    }

    // Returns the total number of indices.
    size_t getLength() const
    {
        return mLength;
    }

private:
    size_t* mIndices; // Array holding the indices [0, 1, ... N-1]
    size_t mLength;   // The length of mIndices (N)
    size_t mCur;      // Current position within mIndices
};

}

#endif

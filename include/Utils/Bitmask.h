#ifndef BITMASK_H
#define BITMASK_H

#include <bitset>
#include <climits>
#include <cstdint>
#include "Rand.h"

namespace opkit
{

template<typename T>
void printBinary(const T& a)
{
    const char* beg = reinterpret_cast<const char*>(&a);
    const char* end = beg + sizeof(a);
    while(beg != end)
        std::cout << std::bitset<CHAR_BIT>(*beg++) << ' ';
}

// T is the raw type (e.g. unsigned char)
// U is the logical type (e.g. double)
template <typename T, typename U>
void printBinary(const T* array, const size_t length)
{
    size_t multiplier = sizeof(U) / sizeof(T);
    for (size_t i = 0; i < length; i += multiplier)
    {
        for (size_t j = 0; j < multiplier; ++j)
            printBinary(array[i + j]);
        cout << '\n';
    }
}

// Used to determine which type the mask should be based on the size of the
// input in bytes. The default is unsigned char, which should always work.
template <size_t S> struct MaskType
{
    using type = uint8_t;
};

template <> struct MaskType<2>
{
    using type = uint16_t;
};

template <> struct MaskType<4>
{
    using type = uint32_t;
};

template <> struct MaskType<8>
{
    using type = uint64_t;
};

// This class represents a general bitmask that can be used for filtering
// arrays of (usually simple) types.
template <class T>
class Bitmask
{
public:

    using BaseType = typename MaskType<sizeof(T)>::type;

    // Create a new bitmask large enough to cover an array of the given size.
    Bitmask(size_t size) :
        mMultiplier(sizeof(T) / sizeof(BaseType)),
        mMask(mMultiplier * size) {}

    // Clear the bitmask. Every element will be masked out when Bitmask::apply
    // is called.
    void clear()
    {
        std::fill(mMask.begin(), mMask.end(), 0);
    }

    // Set an individual element of the bitmask. Those elements that are set
    // will be kept when Bitmask::apply() is called.
    void set(const size_t index)
    {
        for (size_t j = 0; j < mMultiplier; ++j)
            mMask[mMultiplier * index + j] = ~(0);
    }

    // Set all elements of the bitmask. This effectively means the bitmask uses
    // an identity masking. No elements will be masked.
    void setAll()
    {
        std::fill(mMask.begin(), mMask.end(), ~(0));
    }

    // Randomly set a given percentage (between [0.0, 1.0]) of the elements of
    // this mask.
    void setRandom(Rand& rand, const double fillPercentage)
    {
        const size_t length         = mMask.size() / mMultiplier;
        const size_t numConnections = (size_t)(length * fillPercentage);

        // Create a mask from the bottom up
        if (fillPercentage < 0.5)
        {
            for (size_t i = 0; i < numConnections; ++i)
            {
                // Pick random index
                size_t index;
                do
                {
                    index = mMultiplier * rand.nextInteger(0ul, length - 1);
                } while(mMask[index] == ~(0));

                for (size_t j = 0; j < mMultiplier; ++j)
                    mMask[index + j] = ~(0);
            }
        }

        // Create a mask from the top down
        else
        {
            std::fill(mMask.begin(), mMask.end(), ~(0));
            for (size_t i = 0; i < length - numConnections; ++i)
            {
                // Pick random index
                size_t index;
                do
                {
                    index = mMultiplier * rand.nextInteger(0ul, length - 1);
                } while(mMask[index] == 0);

                for (size_t j = 0; j < mMultiplier; ++j)
                    mMask[index + j] = 0;
            }
        }
    }

    // Applies this mask to the given array of data. For each element in which
    // the mask is set, data[i] will be left alone. For each element that has
    // not been set, data[i] will be set to 0.
    void apply(T* data)
    {
        BaseType* ptr  = reinterpret_cast<BaseType*>(data);
        BaseType* mask = mMask.data();

        const size_t length = mMask.size();
        for (size_t i = 0; i < length; ++i)
            ptr[i] &= mask[i];
    }

private:
    // The mask is represented as an array of bytes in order to achieve maximum
    // generality. If it is possible to use a larger datatype, we will, since
    // the mask operation is more efficient that way. The multiplier represents
    // the ratio between the source type and the mask type.
    size_t mMultiplier;
    vector< BaseType > mMask;
};
}

#endif

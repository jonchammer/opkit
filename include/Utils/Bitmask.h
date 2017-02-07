#ifndef BITMASK_H
#define BITMASK_H

#include <bitset>
#include <climits>
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
void printBinary(const T* array, size_t length)
{
    for (size_t i = 0; i < length; i += sizeof(U))
    {
        for (size_t j = 0; j < sizeof(U); ++j)
            printBinary(array[i + j]);
        cout << '\n';
    }
}

// This class represents a general bitmask that can be used for filtering
// arrays of (usually simple) types.
template <class T>
class Bitmask
{
public:

    // Create a new bitmask large enough to cover an array of the given size.
    Bitmask(size_t size) : mMask(sizeof(T) * size) {}

    // Clear the bitmask. Every element will be masked out when Bitmask::apply
    // is called.
    void clear()
    {
        std::fill(mMask.begin(), mMask.end(), 0x00);
    }

    // Set an individual element of the bitmask. Those elements that are set
    // will be kept when Bitmask::apply() is called.
    void set(const size_t index)
    {
        for (size_t j = 0; j < sizeof(T); ++j)
            mMask[sizeof(T) * index + j] = 0xFF;
    }

    // Set all elements of the bitmask. This effectively means the bitmask uses
    // an identity masking. No elements will be masked.
    void setAll()
    {
        std::fill(mMask.begin(), mMask.end(), 0xFF);
    }

    // Randomly set a given percentage (between [0.0, 1.0]) of the elements of
    // this mask.
    void setRandom(Rand& rand, const double fillPercentage)
    {
        const size_t length         = mMask.size() / sizeof(T);
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
                    index = sizeof(T) * rand.nextInteger(0ul, length - 1);
                } while(mMask[index] == 0xFF);

                for (size_t j = 0; j < sizeof(T); ++j)
                    mMask[index + j] = 0xFF;
            }
        }

        // Create a mask from the top down
        else
        {
            std::fill(mMask.begin(), mMask.end(), 0xFF);
            for (size_t i = 0; i < length - numConnections; ++i)
            {
                // Pick random index
                size_t index;
                do
                {
                    index = sizeof(T) * rand.nextInteger(0ul, length - 1);
                } while(mMask[index] == 0x00);

                for (size_t j = 0; j < sizeof(T); ++j)
                    mMask[index + j] = 0x00;
            }
        }
    }

    // Applies this mask to the given array of data. For each element in which
    // the mask is set, data[i] will be left alone. For each element that has
    // not been set, data[i] will be set to 0.
    void apply(T* data)
    {
        unsigned char* ptr  = reinterpret_cast<unsigned char*>(data);
        unsigned char* mask = mMask.data();

        const size_t length = mMask.size();
        for (size_t i = 0; i < length; ++i)
            ptr[i] &= mask[i];
    }

private:
    // The mask is represented as an array of bytes in order to achieve maximum
    // generality.
    vector<unsigned char> mMask;
};
}

#endif

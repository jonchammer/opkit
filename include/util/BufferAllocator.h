#ifndef BUFFER_ALLOCATOR_H
#define BUFFER_ALLOCATOR_H

#include <vector>
#include <algorithm>

namespace BufferAllocatorDetail
{

// This class provides the actual implementation of the BufferAllocator.
// It allocates a large block of memory using new[] and then provides pieces
// to interested objects.
//
// The singleton pattern is used for this class. Rather than using a constructor,
// obtain an instance by calling the static instance() method.
template <class T>
class Buffer
{
private:
    using Range = std::pair<size_t, size_t>;

    // Storage for the actual data
    T* data;

    // An array of [start, end] pairs that dictate which blocks of 'data' are
    // currently available for use
    Range* available;

    // The index of the next available slot in the 'available' array
    size_t availableIndex;

    // Create a new Buffer object, marking the entire range as available
    Buffer(const size_t size) :
        data(new T[size]),
        available(new Range[size]),
        availableIndex(0)
    {
        available[availableIndex++] = {0, size - 1};
    }

public:

    // Obtain an instance to the single Buffer object. 'size' dictates how many
    // T's (not bytes) can be allocated simultaneously.
    static Buffer& instance(const size_t size = 0)
    {
        static Buffer<T> buffer(size);
        return buffer;
    }

    ~Buffer()
    {
        delete[] data;
        delete[] available;
        availableIndex = 0;
    }

    T* allocate(const size_t n)
    {
        // Search for a contiguous piece large enough to hold 'n' elements
        T* res = search(n);
        if (res != nullptr)
            return res;

        // No piece exists that is large enough.
        else
        {
            #ifndef NDEBUG
                std::cerr << "Unable to allocate " << n << " bytes." << std::endl;
                std::cerr << "Available Allocations: " << std::endl;
                print(true);
            #endif
            throw std::bad_alloc();
        }
    }

    void deallocate(T* ptr, const size_t n)
    {
        size_t start = ptr - data;
        size_t end   = start + n - 1;

        // We must add the range for the deallocated memory to the available
        // list (which itself should be kept in sorted order). We will actually
        // create a new element in the worst case, but we will attempt to merge
        // this range with an existing one first for optimal performance.

        // Merge first
        if (end == available[0].first - 1)
        {
            available[0].first = start;
        }
        // Insert first
        else if (end < available[0].first)
        {
            for (int i = availableIndex; i >= 0; --i)
                available[i + 1] = available[i];
            available[0] = {start, end};
            ++availableIndex;
        }
        // Merge last
        else if (start == available[availableIndex - 1].second + 1)
        {
            available[availableIndex - 1].second = end;
        }
        // Insert last
        else if (start > available[availableIndex - 1].second)
        {
            available[availableIndex++] = {start, end};
        }
        // Merge / Insert middle
        else
        {
            auto pair    = std::make_pair(start, end);
            size_t index = std::lower_bound(available,
                available + availableIndex, pair) - available - 1;
            Range& prev = available[index];
            Range& next = available[index + 1];
            if (start == prev.second + 1)
            {
                prev.second = end;
                if (next.first == prev.second + 1)
                {
                    prev.second = next.second;
                    for (size_t i = index + 1; i < availableIndex; ++i)
                        available[i] = available[i + 1];
                    --availableIndex;
                }
            }
            else
            {
                if (end == next.first - 1)
                    next.first = start;
                else
                {
                    for (size_t i = availableIndex; i > index; --i)
                        available[i + 1] = available[i];
                    available[index + 1] = pair;
                    ++availableIndex;
                }
            }
        }
    }

    // Used for debugging
    void print(bool sorted = false)
    {
        if (sorted)
        {
            std::sort(available, available + availableIndex,
            [](const Range& p1, const Range& p2)
            {
                if (p1.first == p2.first)
                    return p1.second < p2.second;
                else return p1.first < p2.first;
            });
        }

        for (size_t i = 0; i < availableIndex; ++i)
        {
            Range& range = available[i];
            std::cout << '[' << range.first << " " << range.second << "]\n";
        }
    }

private:

    // Scan through each of the available ranges for a section large
    // enough to hold 'n' elements. If we find one, return a pointer
    // to that memory and update the available list. Otherwise, return nullptr.
    T* search(const size_t n)
    {
        // static const double alpha = 0.01;
        // static double mean        = 20.0;
        for (int i = availableIndex - 1; i >= 0; --i)
        {
            Range& cur = available[i];
            if (cur.second - cur.first + 1 >= n)
            {
                T* ptr = data + cur.first;
                cur.first += n;
                if (cur.first > cur.second)
                {
                    using namespace std;
                    swap(available[i], available[availableIndex - 1]);
                    --availableIndex;
                }

                // mean = alpha * i + (1.0 - alpha) * mean;
                // std::cout << mean << "\n";
                return ptr;
            }
        }

        return nullptr;
    }
};

}

namespace opkit
{

// User-facing allocator class. T represents the type of the object being
// allocated, and Size is the buffer capacity. At most 'Size' elements may be
// allocated at any point in time.
template <class T, size_t Size = 64>
class BufferAllocator
{
private:
    // Cache the singleton reference
    BufferAllocatorDetail::Buffer<T>& buffer;

public:
    typedef T value_type;

    // Create a new buffer allocator. All instances share the same buffer.
    BufferAllocator() :
        buffer(BufferAllocatorDetail::Buffer<T>::instance(Size)) {}

    template <class U, size_t size>
    BufferAllocator(const BufferAllocator<U, size>&) :
        buffer(BufferAllocatorDetail::Buffer<T>::instance(Size)) {}

    T* allocate(std::size_t n)
    {
        return buffer.allocate(n);
    }

    void deallocate(T* ptr, std::size_t n) noexcept
    {
        buffer.deallocate(ptr, n);
    }

    void print(bool sorted = false)
    {
        buffer.print(sorted);
    }

    template<typename U>
    struct rebind
    {
        typedef BufferAllocator<U, Size> other;
    };
};

template <class T, class U, size_t size1, size_t size2>
bool operator==(const BufferAllocator<T, size1>&, const BufferAllocator<U, size2>&)
{
    return false;
}

template <class T, class U, size_t size1, size_t size2>
bool operator!=(const BufferAllocator<T, size1>&, const BufferAllocator<U, size2>&)
{
    return true;
}

template <class T, size_t size1, size_t size2>
bool operator==(const BufferAllocator<T, size1>&, const BufferAllocator<T, size2>&)
{
    return true;
}

template <class T, size_t size1, size_t size2>
bool operator!=(const BufferAllocator<T, size1>&, const BufferAllocator<T, size2>&)
{
    return false;
}

}
#endif

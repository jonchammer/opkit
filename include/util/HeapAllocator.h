#ifndef HEAP_ALLOCATOR_H
#define HEAP_ALLOCATOR_H

#include <vector>
#include <algorithm>

namespace HeapAllocatorDetail
{

// This class provides the actual implementation of the HeapAllocator.
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

    // A heap of [start, end] pairs that dictate which blocks of 'data' are
    // currently available for use
    Range* available;

    // The index of the next available slot in the 'available' heap
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
        // The top of the heap holds the largest contiguous cell. If it has
        // enough room, use it.
        T* res = search(n);
        if (res != nullptr) return res;

        // No piece exists that is large enough.
        else
        {
            mergeAvailable();
            res = search(n);
            if (res != nullptr) return res;
            else                throw std::bad_alloc();
        }
    }

    void deallocate(T* ptr, const size_t n)
    {
        size_t start = ptr - data;
        size_t end   = start + n - 1;

        // Try the simple optimization of merging with the top of the heap
        Range& max = available[0];
        if (end == max.first - 1)
            max.first = start;
        else if (start == max.second + 1)
            max.second = end;

        // Create a new heap element and fix the heap
        else
        {
            available[availableIndex++] = {start, end};
            std::push_heap(available, available + availableIndex, distanceComp);
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

        for (int i = 0; i < availableIndex; ++i)
        {
            Range& range = available[i];
            std::cout << '[' << range.first << " " << range.second << "]\n";
        }
    }

private:

    // Take memory from the top of the heap when needed, then update the
    // structure to maintain the heap property.
    T* search(const size_t n)
    {
        Range& max = available[0];
        if (max.second - max.first + 1 >= n)
        {
            T* ptr     = data + max.first;
            max.first += n;

            // Update the heap
            if (max.first > max.second)
            {
                std::pop_heap(available, available + availableIndex, distanceComp);
                --availableIndex;
            }
            else shuffleDown(available, availableIndex, 0);

            return ptr;
        }
        else return nullptr;
    }

    // Flatten all the available ranges into the smallest possible units. This
    // is necessary when the memory has become too fragmented.
    void mergeAvailable()
    {
        std::sort(available, available + availableIndex,
        [](const Range& p1, const Range& p2)
        {
            if (p1.first == p2.first)
                return p1.second < p2.second;
            else return p1.first < p2.first;
        });

        // Merge the ranges in place
        int destIndex = 1;
        for (int i = 1; i < availableIndex; ++i)
        {
            Range& prev = available[destIndex - 1];
            Range& cur  = available[i];

            if (cur.first == prev.second + 1)
                prev.second = cur.second;
            else
                available[destIndex++] = cur;
        }
        availableIndex = destIndex;

        // Turn into a heap
        std::make_heap(available, available + availableIndex, distanceComp);
    }

    // Used to restore the heap property after the max element has been
    // updated in allocate().
    void shuffleDown(Range* data, const int size, const int start)
    {
        int parent  = start;
        int largest = -1;
        while (parent != largest)
        {
            largest = parent;

            // Examine the left child
            int left = 2 * parent + 1;
            if (left < size && !distanceComp(data[left], data[largest]))
                largest = left;

            // Examine the right child
            int right = 2 * parent + 2;
            if (right < size && !distanceComp(data[right], data[largest]))
                largest = right;

            // Swap parent with largest child if necessary
            if (parent != largest)
            {
                using namespace std;
                swap(data[parent], data[largest]);
                parent  = largest;
                largest = -1;
            }
        }
    }

    // Compare two ranges based on their distance. Used for the heap operations.
    static bool distanceComp(const Range& A, const Range& B)
    {
        return (A.second - A.first) < (B.second - B.first);
    }
};

}

namespace tensorlib
{

// User-facing allocator class. T represents the type of the object being
// allocated, and Size is the buffer capacity. At most 'Size' elements may be
// allocated at any point in time.
template <class T, size_t Size = 64>
class HeapAllocator
{
private:
    // Cache the singleton reference
    HeapAllocatorDetail::Buffer<T>& buffer;

public:
    typedef T value_type;

    // Create a new buffer allocator. All instances share the same buffer.
    HeapAllocator() :
        buffer(HeapAllocatorDetail::Buffer<T>::instance(Size)) {}

    template <class U, size_t size>
    constexpr HeapAllocator(const HeapAllocator<U, size>&) noexcept {}

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
        typedef HeapAllocator<U, Size> other;
    };
};

template <class T, class U, size_t size1, size_t size2>
bool operator==(const HeapAllocator<T, size1>&, const HeapAllocator<U, size2>&)
{
    return false;
}

template <class T, class U, size_t size1, size_t size2>
bool operator!=(const HeapAllocator<T, size1>&, const HeapAllocator<U, size2>&)
{
    return true;
}

template <class T, size_t size1, size_t size2>
bool operator==(const HeapAllocator<T, size1>&, const HeapAllocator<T, size2>&)
{
    return true;
}

template <class T, size_t size1, size_t size2>
bool operator!=(const HeapAllocator<T, size1>&, const HeapAllocator<T, size2>&)
{
    return false;
}

}
#endif

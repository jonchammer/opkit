#ifndef TENSOR_ITERATOR_H
#define TENSOR_ITERATOR_H

template <class T>
class Tensor;

// This class implements both the constant and non-const Tensor iterators.
// It is optimized to be as fast as possible, delegating to simple pointer
// arithmetic whenever the original tensor is continuous.
//
// NOTE: Internally, a union is used to reduce the memory footprint. Either a
// raw pointer will be stored or a vector containing the current index will be.
// The 'mTag' member is decided at object-instantiation time and controls which
// of the two will be used. Because vectors have non-trivial constructors and
// destructors, we have to explicitly manage the object lifecycle via placement
// new and manual calling of the object destructor.
template <class T, bool isConst = true>
class TensorIterator : public std::iterator< std::forward_iterator_tag, T >
{
public:
    typedef typename std::conditional<isConst, const Tensor<T>*, Tensor<T>*>::type ParentType;
    typedef typename std::conditional<isConst, const T&, T&>::type ReferenceType;
    typedef typename std::conditional<isConst, const T*, T*>::type PointerType;

    friend class TensorIterator<T, true>;

    // Constructors
    TensorIterator(ParentType parent, bool end = false) :
        mParent(parent)
    {
        if (parent->contiguous())
        {
            mTag = SIMPLE;
            mPtr = parent->data();
            if (end) mPtr += parent->size();
        }
        else
        {
            mTag = ADVANCED;
            new(&mIndex) SmallVector(parent->rank());
            if (end) mIndex[0] = parent->shape(0);
        }
    }

    // Copy constructor
    TensorIterator(const TensorIterator<T, isConst>& other) :
        mTag(other.mTag),
        mParent(other.mParent)
    {
        switch (mTag)
        {
            case SIMPLE:
                mPtr = other.mPtr;
            break;

            case ADVANCED:
                new(&mIndex) SmallVector(other.mIndex);
            break;
        }
    }

    // Destructor
    ~TensorIterator()
    {
        if (mTag == ADVANCED)
            mIndex.~SmallVector();
    }

    // Increment / decrement
    TensorIterator& operator++()
    {
        switch (mTag)
        {
            case SIMPLE:
                ++mPtr;
            break;

            case ADVANCED:
                // Update the index (N-ary counter)
                int dim = mIndex.size() - 1;
                mIndex[dim]++;
                while (dim > 0 && mIndex[dim] >= mParent->shape(dim))
                {
                    mIndex[dim] = 0;
                    mIndex[dim - 1]++;
                    --dim;
                }
            break;
        }

        return *this;
    }

    TensorIterator operator++(int)
    {
        TensorIterator ret = *this;
        ++(*this);
        return ret;
    }

    // Comparison
    bool operator==(const TensorIterator& other)
    {
        if (mTag != other.mTag || mParent != other.mParent)
            return false;

        switch (mTag)
        {
            case SIMPLE:   return mPtr   == other.mPtr;
            default:       return mIndex == other.mIndex;
        }
    }

    bool operator!=(const TensorIterator& other)
    {
        return !(*this == other);
    }

    // Assignment
    TensorIterator& operator=(const TensorIterator& other)
    {
        if (this != &other)
        {
            if (mTag == ADVANCED)
                mIndex.~SmallVector();

            if (other.mTag == ADVANCED)
                new (&mIndex) SmallVector(other.mIndex);
            else mPtr = other.mPtr;

            mTag    = other.mTag;
            mParent = other.mParent;
        }

        return *this;
    }

    // Movement
    TensorIterator& operator=(TensorIterator&& other)
    {
        if (this != &other)
        {
            if (mTag == ADVANCED)
                mIndex.~SmallVector();

            if (other.mTag == ADVANCED)
                new (&mIndex) SmallVector(std::move(other.mIndex));
            else mPtr = other.mPtr;

            mTag    = other.mTag;
            mParent = other.mParent;
        }

        return *this;
    }

    // Dereferencing
    ReferenceType operator*()
    {
        switch (mTag)
        {
            case SIMPLE:
                return *mPtr;
            default:
                return mParent->at(mIndex.begin(), mIndex.end());
        }
    }

    // Additional methods
    template <class Alloc>
    void index(vector<size_t, Alloc>& res) const
    {
        switch (mTag)
        {
            case ADVANCED:
                res.assign(mIndex.begin(), mIndex.end());
                return;

            default:
                res.resize(mParent->rank());
                size_t index = mPtr - mParent->data();

                for (int i = res.size() - 1; i >= 0; --i)
                {
                    size_t shape = mParent->shape(i);
                    res[i] = index % shape;
                    index /= shape;
                }
                return;
        }
    }

private:

    enum {SIMPLE, ADVANCED} mTag;
    ParentType mParent;

    union
    {
        PointerType mPtr;
        SmallVector mIndex;
    };
};

#endif

#ifndef REFERENCE_COUNT_H
#define REFERENCE_COUNT_H

namespace tensorlib
{

// See: http://www.aristeia.com/BookErrata/M29Source.html
class RCObject
{                                      // base class for reference-
public:                                // counted objects
    void addReference();
    void removeReference();
    void markUnshareable();
    bool isShareable() const;
    bool isShared() const;

protected:
    RCObject();
    RCObject(const RCObject& rhs);
    RCObject& operator=(const RCObject& rhs);
    RCObject(RCObject&& rhs);
    RCObject& operator=(RCObject&& rhs);
    virtual ~RCObject() = 0;

private:
    int refCount;
    bool shareable;
};

RCObject::RCObject() :
    refCount(0), shareable(true) {}

RCObject::RCObject(const RCObject&) :
    refCount(0), shareable(true) {}

RCObject& RCObject::operator=(const RCObject&)
{
    return *this;
}

RCObject::RCObject(RCObject&& rhs) :
    refCount(rhs.refCount), shareable(rhs.shareable)
{
    rhs.refCount  = 0;
    rhs.shareable = false;
}

RCObject& RCObject::operator=(RCObject&& rhs)
{
    refCount      = rhs.refCount;
    shareable     = rhs.shareable;
    rhs.refCount  = 0;
    rhs.shareable = false;
    return *this;
}

RCObject::~RCObject() {}

void RCObject::addReference()
{
    ++refCount;
}

void RCObject::removeReference()
{
    if (--refCount == 0) delete this;
}

void RCObject::markUnshareable()
{
    shareable = false;
}

bool RCObject::isShareable() const
{
    return shareable;
}

bool RCObject::isShared() const
{
    return refCount > 1;
}


/******************************************************************************
*                 Template Class RCPtr (from pp. 203, 206)                    *
******************************************************************************/
template <class T>                     // template class for smart
class RCPtr                            // pointers-to-T objects; T
{
public:                                // must support the RCObject interface
    RCPtr(T* realPtr = nullptr);
    RCPtr(const RCPtr& rhs);
    RCPtr(RCPtr&& rhs) noexcept;
    ~RCPtr() noexcept;
    RCPtr& operator=(const RCPtr& rhs);
    RCPtr& operator=(RCPtr&& rhs) noexcept;

    bool operator==(const RCPtr& other) const;
    bool operator!=(const RCPtr& other) const;
    bool operator==(const T* other) const;
    bool operator!=(const T* other) const;

    T* operator->() const;
    T& operator*() const;

private:
    T *pointee;
    void init();
};

template <class T>
void RCPtr<T>::init()
{
    if (pointee == nullptr) return;

    if (!pointee->isShareable())
        pointee = pointee->clone(); //new T(*pointee);

    pointee->addReference();
}

template <class T>
RCPtr<T>::RCPtr(T* realPtr)
    : pointee(realPtr)
{
    init();
}

template <class T>
RCPtr<T>::RCPtr(const RCPtr& rhs)
    : pointee(rhs.pointee)
{
    init();
}

template <class T>
RCPtr<T>::RCPtr(RCPtr&& rhs) noexcept
    : pointee(rhs.pointee)
{
    rhs.pointee = nullptr;
}

template <class T>
RCPtr<T>::~RCPtr() noexcept
{
    if (pointee != nullptr) pointee->removeReference();
}

template <class T>
RCPtr<T>& RCPtr<T>::operator=(const RCPtr& rhs)
{
    if (pointee != rhs.pointee)
    {
        T* oldPointee = pointee;

        pointee = rhs.pointee;
        init();

        if (oldPointee != nullptr) oldPointee->removeReference();
    }

    return *this;
}

template <class T>
RCPtr<T>& RCPtr<T>::operator=(RCPtr&& rhs) noexcept
{
    if (pointee != nullptr) pointee->removeReference();
    pointee     = rhs.pointee;
    rhs.pointee = nullptr;
    return *this;
}

template <class T>
bool RCPtr<T>::operator==(const RCPtr& other) const
{
    return pointee == other.pointee;
}

template <class T>
bool RCPtr<T>::operator!=(const RCPtr& other) const
{
    return !(*this == other);
}

template <class T>
bool RCPtr<T>::operator==(const T* other) const
{
    return pointee == other;
}

template <class T>
bool RCPtr<T>::operator!=(const T* other) const
{
    return !(*this == other);
}

template <class T>
T* RCPtr<T>::operator->() const
{
    return pointee;
}

template <class T>
T& RCPtr<T>::operator*() const
{
    return *pointee;
}
}

#endif

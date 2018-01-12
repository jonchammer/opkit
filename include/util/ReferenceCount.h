#ifndef REFERENCE_COUNT_H
#define REFERENCE_COUNT_H

namespace opkit
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
    int getRefCount() const;

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
    if (--refCount == 0)
        delete this;
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

int RCObject::getRefCount() const
{
    return refCount;
}

/******************************************************************************
*                 Template Class RCPtr (from pp. 203, 206)                    *
******************************************************************************/
template <class T>                     // template class for smart
class RCPtr                            // pointers-to-T objects; T
{
public:                                // must support the RCObject interface
    RCPtr(T* realPtr = nullptr, bool weak = false);
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
    bool weak;    // Weak pointers do not update the reference count
    void init();
};

template <class T>
void RCPtr<T>::init()
{
    if (pointee == nullptr) return;

    if (!pointee->isShareable())
        pointee = pointee->clone(); //new T(*pointee);

    if (!weak) pointee->addReference();
}

template <class T>
RCPtr<T>::RCPtr(T* realPtr, bool weak)
    : pointee(realPtr), weak(weak)
{
    init();
}

template <class T>
RCPtr<T>::RCPtr(const RCPtr& rhs)
    : pointee(rhs.pointee), weak(rhs.weak)
{
    init();
}

template <class T>
RCPtr<T>::RCPtr(RCPtr&& rhs) noexcept
    : pointee(rhs.pointee), weak(rhs.weak)
{
    rhs.pointee = nullptr;
    rhs.weak    = false;
}

template <class T>
RCPtr<T>::~RCPtr() noexcept
{
    if (pointee != nullptr && !weak) pointee->removeReference();
}

template <class T>
RCPtr<T>& RCPtr<T>::operator=(const RCPtr& rhs)
{
    if (pointee != rhs.pointee)
    {
        T* oldPointee = pointee;
        bool oldWeak  = weak;

        pointee = rhs.pointee;
        weak    = rhs.weak;
        init();

        if (oldPointee != nullptr && !oldWeak) oldPointee->removeReference();
    }

    return *this;
}

template <class T>
RCPtr<T>& RCPtr<T>::operator=(RCPtr&& rhs) noexcept
{
    if (pointee != nullptr && !weak) pointee->removeReference();
    pointee     = rhs.pointee;
    weak        = rhs.weak;
    rhs.pointee = nullptr;
    rhs.weak    = false;
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

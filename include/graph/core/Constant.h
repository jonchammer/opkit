#ifndef CONSTANT_H
#define CONSTANT_H

#include "tensor/Tensor.h"
#include "tensor/TensorOps.h"

namespace opkit
{

// A graph node that acts as a placeholder for a single Tensor that cannot be
// modified later.
template <class T>
class Constant : public Node<T>
{
protected:
    Tensor<T> mValue;
    std::string mName;

public:
    // Normal constructors
    Constant()                        = delete;
    Constant(const Constant<T>& orig) = default;
    Constant(Constant<T>&& orig)      = default;

    // Construct and initialize simultaneously
    template <class TensorType>
    Constant(const std::string& name, TensorType&& val) :
        mValue(std::forward<TensorType>(val)), mName(name) {}
    Constant(const std::string& name) :
        mValue(zeroes<T>({1})), mName(name) {}

    // Assignment operators
    Constant& operator=(const Constant& orig) = default;
    Constant& operator=(Constant&& orig)      = default;

    // Node class implementations
    const Tensor<T>& operator()() override
    {
        return mValue;
    }

    std::string name() const override
    {
        return mName;
    }

    std::ostream& print(std::ostream& out) const override
    {
        out << name();
        return out;
    }

    Constant* clone() const override
    {
        return new Constant(*this);
    }

    // Constant-specific functions
    const Tensor<T>& value() const { return mValue; }
};

template <class T, class TensorType>
Graph<T> make_constant(const std::string& name, TensorType&& tensor)
{
    return Graph<T>(new Constant<T>(name,
        std::forward<TensorType>(tensor)), Graph<T>::Type::CONSTANT);
}

template <class T, class ValueType>
Graph<T> make_constant(const ValueType constant)
{
    return Graph<T>(new Constant<T>(std::to_string(constant),
        Tensor<T>::fromScalar(constant)), Graph<T>::Type::CONSTANT);
}

template <class T>
Graph<T> make_constant(const std::string& name)
{
    return Graph<T>(new Constant<T>(name), Graph<T>::Type::CONSTANT);
}

}

#endif

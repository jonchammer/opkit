#ifndef VARIABLE_H
#define VARIABLE_H

#include "tensor/Tensor.h"
#include "tensor/TensorOps.h"

namespace opkit
{

// A graph node that acts as a placeholder for a single Tensor that can be
// modified later.
template <class T>
struct Variable : public Constant<T>
{
    // Normal constructors
    Variable()                        = delete;
    Variable(const Variable<T>& orig) = default;
    Variable(Variable<T>&& orig)      = default;

    // Construct and initialize simultaneously
    template <class TensorType>
    Variable(const std::string& name, TensorType&& val) :
        Constant<T>(name, std::forward<TensorType>(val)) {}
    Variable(const std::string name) :
        Constant<T>(name, zeroes<T>({1})) {}

    // Assignment operators
    Variable& operator=(const Variable& orig) = default;
    Variable& operator=(Variable&& orig)      = default;

    // Node class implementations
    void assign(const Tensor<T>& newValue) override
    {
        this->mValue = newValue;
    }

    Variable* clone() const override
    {
        return new Variable(*this);
    }

    // Variable-specific functions
    Tensor<T>& value() { return this->mValue; }
};

template <class T, class TensorType>
Graph<T> make_variable(const std::string& name, TensorType&& tensor)
{
    return Graph<T>(new Variable<T>(name,
        std::forward<TensorType>(tensor)), Graph<T>::Type::VAR);
}

template <class T>
Graph<T> make_variable(const std::string& name)
{
    return Graph<T>(new Variable<T>(name), Graph<T>::Type::VAR);
}

}

#endif

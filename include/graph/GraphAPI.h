#ifndef GRAPH_API_H
#define GRAPH_API_H

#include "graph/Graph.h"

// This file contains functions that are used to create the computation graphs.
// These can be used by users when defining their own operations, but most of
// the time, the user will call one of the Graph Operations that uses these
// internally.
namespace opkit
{

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

template <class T, class Func>
Graph<T> make_unary(const std::string& name, Func&& func, Graph<T> dependent)
{
    Graph<T> res(new UnaryFunction<T>(name, std::forward<Func>(func),
        dependent), Graph<T>::Type::UNARY);

    dependent.addChild(res);
    return res;
}

template <class T, class Func>
Graph<T> make_binary(const std::string& name, Func&& func,
    Graph<T> dependent1, Graph<T> dependent2)
{
    Graph<T> res(new BinaryFunction<T>(name, std::forward<Func>(func),
        dependent1, dependent2), Graph<T>::Type::BINARY);

    dependent1.addChild(res);
    dependent2.addChild(res);
    return res;
}

template <class T, class VecType>
Graph<T> make_list(VecType&& dependents)
{
    Graph<T> res(new ListNode<T>("list", std::forward<VecType>(dependents)),
        Graph<T>::Type::LIST);

    for (auto& elem : dependents)
        elem.addChild(res);
    return res;
}

template <class T, class Func>
Graph<T> make_update(const std::string& name, Func&& func,
    Graph<T> target, Graph<T> value)
{
    Graph<T> res(new UpdateNode<T>(name, std::forward<Func>(func),
        target, value), Graph<T>::Type::UPDATE);

    target.addChild(res);
    value.addChild(res);
    return res;
}

template <class T, class Func>
Graph<T> make_update(const std::string& name, Func&& func,
    Graph<T> target, Graph<T> value, Graph<T> arg)
{
    Graph<T> res(new UpdateNodeArg<T>(name, std::forward<Func>(func),
        target, value, arg), Graph<T>::Type::UPDATE_ARG);

    target.addChild(res);
    value.addChild(res);
    arg.addChild(res);
    return res;
}

// Creates a new graph node that performs the same task as 'unary', with a
// different parent element.
template <class T>
Graph<T> copyUnary(const Graph<T>& orig, Graph<T> parent)
{
    auto fn = ((UnaryFunction<T>&)(orig.node())).getFunction();
    Graph<T> res(new UnaryFunction<T>(orig.name(), fn, parent), Graph<T>::Type::UNARY);

    parent.addChild(res);
    return res;
}

// Creates a new graph node that performs the same task as 'binary', with
// different parent elements.
template <class T>
Graph<T> copyBinary(const Graph<T>& orig, Graph<T> parent1, Graph<T> parent2)
{
    auto fn = ((BinaryFunction<T>&)(orig.node())).getFunction();
    Graph<T> res(new BinaryFunction<T>(orig.name(), fn, parent1, parent2), Graph<T>::Type::BINARY);

    parent1.addChild(res);
    parent2.addChild(res);
    return res;
}

// Creates a new graph node that performs the same task as this update rule,
// with different parent elements
template <class T>
Graph<T> copyUpdate(const Graph<T>& orig, Graph<T> target, Graph<T> value)
{
    auto fn = ((UpdateNode<T>&)(orig.node())).getFunction();
    Graph<T> res(new UpdateNode<T>(orig.name(), fn, target, value), Graph<T>::Type::UPDATE);

    target.addChild(res);
    value.addChild(res);
    return res;
}

template <class T>
Graph<T> copyUpdate(const Graph<T>& orig, Graph<T> target, Graph<T> value, Graph<T> arg)
{
    auto fn = ((UpdateNodeArg<T>&)(orig.node())).getFunction();
    Graph<T> res(new UpdateNodeArg<T>(orig.name(), fn, target, value, arg), Graph<T>::Type::UPDATE_ARG);

    target.addChild(res);
    value.addChild(res);
    arg.addChild(res);
    return res;
}

}
#endif

#ifndef NN_H
#define NN_H

#include "graph/Graph.h"
#include "graph/ops/GraphOps_all.h"

namespace opkit
{

// -------------------------- Activation Functions -------------------------- //

template <class T>
Graph<T> relu(const Graph<T>& x)
{
    return max(x, make_constant<T>("0"));
}

template <class T>
void dLogistic(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    using namespace std;

    // NOTE: For some reason, the simple formulation commented out below fails
    // to calculate the correct answer, but making a new node does seem to work
    // correctly. TODO: Work out why.
    auto deriv = make_unary<T>("dLogistic", [](const Tensor<T>& A)
    {
        return elementwiseFunc(A, [](const T x)
        {
            return x * (1 - x);
        });
    }, node);

    gradients.push_back( deriv * delta );

    // Doesn't work?
    // gradients.push_back( (node * (T{1} - node)) * delta );
}

template <class T>
Graph<T> logistic(const Graph<T>& x)
{
    registerDerivative<T>("logistic",
        [](const Graph<T>& node, const Graph<T>& delta,
        std::vector<Graph<T>>& gradients) {dLogistic(node, delta, gradients);});

    return make_unary<T>("logistic", [](const Tensor<T>& A)
    {
        return elementwiseFunc(A, [](const T x)
        {
            return T{1} / (T{1} + std::exp(-x));
        });
    }, x);
}

template <class T>
void dSoftplus(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    gradients.push_back( (T{1} / (T{1} + exp(-node.getChild(0)))) * delta );
}

template <class T>
Graph<T> softplus(const Graph<T>& x)
{
    registerDerivative<T>("softplus",
        [](const Graph<T>& node, const Graph<T>& delta,
        std::vector<Graph<T>>& gradients) {dSoftplus(node, delta, gradients);});

    return make_unary<T>("softplus", [](const Tensor<T>& A)
    {
        return elementwiseFunc(A, [](const T x)
        {
            return std::log(T{1} + std::exp(x));
        });
    }, x);
}

template <class T>
void dBentIdentity(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    const Graph<T>& x = node.getChild(0);
    gradients.push_back( (x / (T{2} * sqrt(square(x) + T{1})) + T{1}) * delta );
}

template <class T>
Graph<T> bentIdentity(const Graph<T>& x)
{
    registerDerivative<T>("bentIdentity",
        [](const Graph<T>& node, const Graph<T>& delta,
        std::vector<Graph<T>>& gradients) {dBentIdentity(node, delta, gradients);});

    return make_unary<T>("bentIdentity", [](const Tensor<T>& A)
    {
        return elementwiseFunc(A, [](const T x)
        {
            return T{0.5} * (std::sqrt(x * x + T{1}) - T{1}) + x;
        });
    }, x);
}

template <class T>
Graph<T> softmax(const Graph<T>& x)
{
    Graph<T> shape = rank(x) - T{1};
    Graph<T> temp  = exp(x - reduceMax(x, shape));
    return temp / reduceSum(temp, shape);
}

// --------------------------------- Layers --------------------------------- //

// Simple linear layer: y = x * w + b
template <class T>
Graph<T> linear(const Graph<T>& x, const Graph<T>& w, const Graph<T>& b)
{
    return matrixMultiply(x, w) + b;
}

}

#endif

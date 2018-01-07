#ifndef NN_H
#define NN_H

#include "graph/Graph.h"
#include "graph/ops/GraphOps_all.h"

namespace opkit
{

// -------------------------- Activation Functions -------------------------- //

template <class T>
Graph<T> relu(Graph<T> x)
{
    return max(x, 0);
}

template <class T>
void dLogistic(Graph<T> node, Graph<T> delta, std::vector<Graph<T>>& gradients)
{
    using namespace std;

    // NOTE: For some reason, the simple formulation commented out below fails
    // to calculate the correct answer, but making a new node does seem to work
    // correctly. TODO: Work out why.
    auto deriv = make_unary<T>("dLogistic", [](Tensor<T>& y, const Tensor<T>& A)
    {
        return elementwiseFunc(y, A, [](const T x)
        {
            return x * (1 - x);
        });
    }, node);

    gradients.push_back( deriv * delta );

    // Doesn't work?
    // gradients.push_back( (node * (T{1} - node)) * delta );
}

template <class T>
Graph<T> logistic(Graph<T> x)
{
    registerDerivative<T>("logistic",
        [](Graph<T> node, Graph<T> delta,
        std::vector<Graph<T>>& gradients) {dLogistic(node, delta, gradients);});

    return make_unary<T>("logistic", [](Tensor<T>& y, const Tensor<T>& A)
    {
        return elementwiseFunc(y, A, [](const T x)
        {
            return T{1} / (T{1} + std::exp(-x));
        });
    }, x);
}

template <class T>
void dSoftplus(Graph<T> node, Graph<T> delta, std::vector<Graph<T>>& gradients)
{
    gradients.push_back( (1 / (1 + exp(-node.getChild(0)))) * delta );
}

template <class T>
Graph<T> softplus(Graph<T> x)
{
    registerDerivative<T>("softplus",
        [](Graph<T> node, Graph<T> delta,
        std::vector<Graph<T>>& gradients) {dSoftplus(node, delta, gradients);});

    return make_unary<T>("softplus", [](Tensor<T>& y, const Tensor<T>& A)
    {
        return elementwiseFunc(y, A, [](const T x)
        {
            return std::log(T{1} + std::exp(x));
        });
    }, x);
}

template <class T>
void dBentIdentity(Graph<T> node, Graph<T> delta, std::vector<Graph<T>>& gradients)
{
    Graph<T> x = node.getChild(0);
    gradients.push_back( (x / (2 * sqrt(square(x) + 1)) + 1) * delta );
}

template <class T>
Graph<T> bentIdentity(Graph<T> x)
{
    registerDerivative<T>("bentIdentity",
        [](Graph<T> node, Graph<T> delta,
        std::vector<Graph<T>>& gradients) {dBentIdentity(node, delta, gradients);});

    return make_unary<T>("bentIdentity", [](Tensor<T>& y, const Tensor<T>& A)
    {
        return elementwiseFunc(y, A, [](const T x)
        {
            return T{0.5} * (std::sqrt(x * x + T{1}) - T{1}) + x;
        });
    }, x);
}

template <class T>
Graph<T> softmax(Graph<T> x)
{
    Graph<T> shape = rank(x) - 1;
    Graph<T> temp  = exp(x - reduceMax(x, shape));
    return temp / reduceSum(temp, shape);
}

// --------------------------------- Layers --------------------------------- //

// Simple linear layer: y = x * w + b
template <class T>
Graph<T> linear(Graph<T> x, Graph<T> w, Graph<T> b)
{
    return matrixMultiply(x, w) + b;
}

}

#endif

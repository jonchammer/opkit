#ifndef COST_FUNCTIONS_H
#define COST_FUNCTIONS_H

#include "graph/Graph.h"
#include "graph/ops/GraphOps_all.h"

namespace opkit
{

template <class T>
Graph<T> sse(const Graph<T>& model, const Graph<T>& labels)
{
    return reduceSum(square(model - labels));
}

template <class T>
Graph<T> mse(const Graph<T>& model, const Graph<T>& labels)
{
    return reduceMean(square(model - labels));
}

// General purpose cross-entropy. It doesn't matter what comes before this node
// in the graph, but the user must ensure that the input is > 0 for all values.
template <class T>
Graph<T> crossEntropy(const Graph<T>& model, const Graph<T>& labels)
{
    // A very small number is added to the input to prevent log(0) from
    // becomming NaN.
    Graph<T> axes = rank(labels) - 1;
    Graph<T> temp = labels * log(model + std::numeric_limits<T>::epsilon());
    return reduceMean(-reduceSum(temp, axes));
}

template <class T>
void dSoftmaxCrossEntropy(const Graph<T>& node, const Graph<T>& delta, std::vector<Graph<T>>& gradients)
{
    Graph<T> y      = softmax(node.getChild(0));
    Graph<T> labels = node.getChild(1);

    gradients.push_back( (y - labels) * delta );
    gradients.push_back( make_constant<T>(0) );
}

// Specialized operation for the softmax -> cross entropy case. This function
// is more numerically stable than using crossEntropy(softmax(model)), so it
// should be used whenever possible.
template <class T>
Graph<T> softmaxCrossEntropy(const Graph<T>& model, const Graph<T>& labels)
{
    registerDerivative<T>("softmaxCrossEntropy",
        [](const Graph<T>& node, const Graph<T>& delta,
        std::vector<Graph<T>>& gradients) {dSoftmaxCrossEntropy(node, delta, gradients);});

    // TODO: Add some Tensor operators so this looks less evil.
    Graph<T> temp = make_binary<T>("softmaxCrossEntropy", [](Tensor<T>& y, const Tensor<T>& a, const Tensor<T>& b)
    {
        Tensor<T> axes = Tensor<T>::fromScalar(b.rank() - 1);

        // e^a / ||e^a||
        elementwiseFunc(y, a, [](const T x) { return std::exp(x); });
        divBy(y, reduceSum(y, axes));

        // log(e^a / ||e^a|| + epsilon)
        y.apply([](const T& elem)
        {
            return std::log(elem + std::numeric_limits<T>::epsilon());
        });

        // sum(b * log(e^a / ||e^a|| + epsilon))
        reduceSum_v2(y, multiply(b, y), axes);

        // negation
        y.apply([](const T& elem)
        {
            return -elem;
        });

        return y;
    }, model, labels);

    return reduceMean(temp);
}

template <class T>
Graph<T> sigmoidCrossEntropy(const Graph<T>& model, const Graph<T>& labels)
{
    return max(model, 0) - model * labels + log(1 + exp(-abs(model)));
}

// Returns the number of elements that were correctly classified.
template <class T>
Graph<T> correctCount(const Graph<T>& model, const Graph<T>& labels, bool oneHotLabels = false)
{
    if (oneHotLabels)
        return reduceSum(equal(argmax(model, 1), labels));
    else return reduceSum(equal(argmax(model, 1), argmax(labels, 1)));
}

// Returns the percentage of elements that were classified correctly.
template <class T>
Graph<T> accuracy(const Graph<T>& model, const Graph<T>& labels, bool oneHotLabels = false)
{
    if (oneHotLabels)
        return reduceMean(equal(argmax(model, 1), labels));
    else return reduceMean(equal(argmax(model, 1), argmax(labels, 1)));
}

// Returns the raw number of misclassifications
template <class T>
Graph<T> missCount(const Graph<T>& model, const Graph<T>& labels, bool oneHotLabels = false)
{
    if (oneHotLabels)
        return reduceSum(notEqual(argmax(model, 1), labels));
    else return reduceSum(notEqual(argmax(model, 1), argmax(labels, 1)));
}

// Returns the average number of misclassifications. This should always be [0, 1].
template <class T>
Graph<T> missRate(const Graph<T>& model, const Graph<T>& labels, bool oneHotLabels = false)
{
    if (oneHotLabels)
        return reduceMean(notEqual(argmax(model, 1), labels));
    else return reduceMean(notEqual(argmax(model, 1), argmax(labels, 1)));
}

}
#endif

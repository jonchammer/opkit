#ifndef COST_FUNCTIONS_H
#define COST_FUNCTIONS_H

#include <vector>
#include "graph/core/GraphAPI.h"
#include "graph/ops/GraphOps_all.h"
#include "tensor/TensorMath.h"

namespace detail
{

template <class T>
void softmaxCrossEntropyOneHot(Tensor<T>& res, const Tensor<T>& model, const Tensor<T>& labels)
{
    ASSERT(model.rank() == 2 && labels.rank() == 2, "Ranks must be 2.");
    ASSERT(model.shape(0) == labels.shape(0), "Batch sizes must match.");
    ASSERT(labels.shape(1) == 1, "Labels must be one-hot encoded.");

    const size_t M             = model.shape(0);
    const size_t N             = model.shape(1);
    const size_t modelStrideX  = model.stride(1);
    const size_t modelStrideY  = model.stride(0);
    const size_t labelsStrideY = labels.stride(0);

    res.resize({M, 1});

    const T* modelData  = model.data();
    const T* labelsData = labels.data();
          T* resData    = res.data();

    for (size_t y = 0; y < M; ++y)
    {
        const T* modelRow  = modelData;
        const T* labelsRow = labelsData;

        // Calculate the value for this row. The original formula has been
        // rearranged a bit to remove redundant or unnecessary computations.
        T sum = T{};
        for (size_t x = 0; x < N; ++x)
        {
            sum      += opkit::fastExp(*modelRow);
            modelRow += modelStrideX;
        }
        sum        = std::log(sum + std::numeric_limits<T>::epsilon());
        resData[y] = sum - *(modelData + ((size_t) *labelsRow) * modelStrideX);

        // Move on to the next row
        modelData  += modelStrideY;
        labelsData += labelsStrideY;
    }
}

template <class T>
void softmaxCrossEntropySlow(Tensor<T>& res, const Tensor<T>& model, const Tensor<T>& labels)
{
    std::vector<size_t> axes({labels.rank() - 1});

    // e^a / ||e^a||
    elementwiseFunc(res, model, [](const T x) { return opkit::fastExp(x); });
    divBy(res, reduceSum(res, axes));

    // log(e^a / ||e^a|| + epsilon)
    res.apply([](const T& elem)
    {
        return std::log(elem + std::numeric_limits<T>::epsilon());
    });

    // sum(b * log(e^a / ||e^a|| + epsilon))
    reduceSum(res, multiply(labels, res), axes);

    // negation
    res.apply([](const T& elem)
    {
        return -elem;
    });
}

}

namespace opkit
{

template <class T>
Graph<T> sse(Graph<T> model, Graph<T> labels)
{
    return reduceSum(square(model - labels));
}

template <class T>
Graph<T> mse(Graph<T> model, Graph<T> labels)
{
    return reduceMean(square(model - labels));
}

// General purpose cross-entropy. It doesn't matter what comes before this node
// in the graph, but the user must ensure that the input is > 0 for all values.
template <class T>
Graph<T> crossEntropy(Graph<T> model, Graph<T> labels)
{
    // A very small number is added to the input to prevent log(0) from
    // becomming NaN.
    Graph<T> axes = rank(labels) - 1;
    Graph<T> temp = labels * log(model + std::numeric_limits<T>::epsilon());
    return reduceMean(-reduceSum(temp, axes));
}

template <class T>
void dSoftmaxCrossEntropyFast(Graph<T> node, Graph<T> delta, std::vector<Graph<T>>& gradients)
{
    Graph<T> y      = softmax(node.getParent(0));
    Graph<T> labels = node.getParent(1);

    auto op = make_binary<T>("softmaxEmbeddingOp",
    [](Tensor<T>& res, const Tensor<T>& model, const Tensor<T>& label)
    {
        model.copy(res);
        for (size_t y = 0; y < model.shape(0); ++y)
        {
            size_t index = label.at({y, 0});
            res.at({y, index}) -= T{1};
        }
    }, y, labels);

    gradients.push_back( op * delta );
    gradients.push_back( make_constant<T>(0) );
}

template <class T>
void dSoftmaxCrossEntropySlow(Graph<T> node, Graph<T> delta, std::vector<Graph<T>>& gradients)
{
    Graph<T> y      = softmax(node.getParent(0));
    Graph<T> labels = node.getParent(1);

    gradients.push_back( (y - labels) * delta );
    gradients.push_back( make_constant<T>(0) );
}

// Specialized operation for the softmax -> cross entropy case. This function
// is more numerically stable than using crossEntropy(softmax(model)), so it
// should be used whenever possible.
template <class T>
Graph<T> softmaxCrossEntropy(Graph<T> model, Graph<T> labels, bool oneHotLabels = false)
{
    if (oneHotLabels)
    {
        registerDerivative<T>("softmaxCrossEntropyFast",
        [](Graph<T> node, Graph<T> delta,
            std::vector<Graph<T>>& gradients)
        {
            dSoftmaxCrossEntropyFast(node, delta, gradients);
        });

        Graph<T> temp = make_binary<T>("softmaxCrossEntropyFast",
        [](Tensor<T>& res, const Tensor<T>& a, const Tensor<T>& b)
        {
            detail::softmaxCrossEntropyOneHot(res, a, b);
        }, model, labels);

        return reduceMean(temp);
    }

    else
    {
        registerDerivative<T>("softmaxCrossEntropySlow",
        [](Graph<T> node, Graph<T> delta,
            std::vector<Graph<T>>& gradients)
        {
            dSoftmaxCrossEntropySlow(node, delta, gradients);
        });

        Graph<T> temp = make_binary<T>("softmaxCrossEntropySlow",
        [](Tensor<T>& res, const Tensor<T>& a, const Tensor<T>& b)
        {
            detail::softmaxCrossEntropySlow(res, a, b);
        }, model, labels);
        return reduceMean(temp);
    }
}

template <class T>
Graph<T> sigmoidCrossEntropy(Graph<T> model, Graph<T> labels)
{
    return max(model, 0) - model * labels + log(1 + exp(-abs(model)));
}

// Returns the number of elements that were correctly classified.
template <class T>
Graph<T> correctCount(Graph<T> model, Graph<T> labels, bool oneHotLabels = false)
{
    if (oneHotLabels)
        return reduceSum(equal(argmax(model, 1), labels));
    else return reduceSum(equal(argmax(model, 1), argmax(labels, 1)));
}

// Returns the percentage of elements that were classified correctly.
template <class T>
Graph<T> accuracy(Graph<T> model, Graph<T> labels, bool oneHotLabels = false)
{
    if (oneHotLabels)
        return reduceMean(equal(argmax(model, 1), labels));
    else return reduceMean(equal(argmax(model, 1), argmax(labels, 1)));
}

// Returns the raw number of misclassifications
template <class T>
Graph<T> missCount(Graph<T> model, Graph<T> labels, bool oneHotLabels = false)
{
    if (oneHotLabels)
        return reduceSum(notEqual(argmax(model, 1), labels));
    else return reduceSum(notEqual(argmax(model, 1), argmax(labels, 1)));
}

// Returns the average number of misclassifications. This should always be [0, 1].
template <class T>
Graph<T> missRate(Graph<T> model, Graph<T> labels, bool oneHotLabels = false)
{
    if (oneHotLabels)
        return reduceMean(notEqual(argmax(model, 1), labels));
    else return reduceMean(notEqual(argmax(model, 1), argmax(labels, 1)));
}

}
#endif

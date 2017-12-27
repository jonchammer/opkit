// Comment to disable debug assertions
// #define NDEBUG

#include <iostream>
#include <unordered_map>
#include <set>
#include <vector>

#include "opkit/opkit.h"

using namespace std;
using namespace opkit;

template <class T>
Tensor<T> convertColumnToOneHot(const Tensor<T>& input, const size_t column)
{
    ASSERT(input.rank() == 2, "Only 2D matrices are currently supported.");
    const size_t rows = input.shape(0);
    const size_t cols = input.shape(1);

    // Calculate the number of unique values in this column
    std::set<double> elements;
    for (size_t r = 0; r < rows; ++r)
        elements.insert(input.at({r, column}));

    // Create the resulting tensor
    Tensor<T> result({rows, cols + elements.size() - 1});
    result.fill(T{});
    for (size_t r = 0; r < rows; ++r)
    {
        // Copy the data before the column
        size_t c = 0;
        for (; c < column; ++c)
            result.at({r, c}) = input.at({r, c});

        // Convert the column itself
        size_t val = input.at({r, column});
        result.at({r, c + val}) = T{1};

        // Copy the data after the column
        for(++c; c < cols; ++c)
            result.at({r, c + elements.size()}) = input.at({r, c});
    }
    return result;
}

int main()
{
    using T = float;

    // Load the training and testing data
    Tensor<T> trainFeatures, trainLabels, testFeatures, testLabels;
    loadTensorRaw("/home/jhammer/data/mnist/mnist_train_features_float.raw", trainFeatures);
    loadTensorRaw("/home/jhammer/data/mnist/mnist_train_labels_float.raw",   trainLabels);
    loadTensorRaw("/home/jhammer/data/mnist/mnist_test_features_float.raw",  testFeatures);
    loadTensorRaw("/home/jhammer/data/mnist/mnist_test_labels_float.raw",    testLabels);

    // Perform preprocessing
    trainFeatures.apply([](const T& x) { return x / T{255}; });
    testFeatures.apply([](const T& x) { return x / T{255}; });
    trainLabels = convertColumnToOneHot(trainLabels, 0);
    testLabels  = convertColumnToOneHot(testLabels, 0);

    cout << "\% Train features: " << trainFeatures.shape() << endl;
    cout << "\% Train labels:   " << trainLabels.shape()   << endl;
    cout << "\% Test features:  " << testFeatures.shape()  << endl;
    cout << "\% Test labels:    " << testLabels.shape()    << endl
         << "\%\n";

    // Construct the variables
    auto x1 = make_variable<T>("x", trainFeatures);
    auto y1 = make_variable<T>("y", trainLabels);
    auto w1 = make_variable<T>("w1", normal<T>({784, 10}, 0, 0.01, 42));
    auto b1 = make_variable<T>("b1", normal<T>({1, 10}, 0, 0.01, 42));

    // Build the graph with error functions
    auto y     = linear(x1, w1, b1);
    auto error = softmaxCrossEntropy(y, y1);
    auto miss  = missCount(y, y1, false);

    // Build the update rule
    auto update = gradientDescent(error, {"w1", "b1"}, T{0.1});

    printf("@RELATION network_training\n");
    printf("@ATTRIBUTE Epoch real\n");
    printf("@ATTRIBUTE Cumulative_Time_Seconds real\n");
    printf("@ATTRIBUTE Misclassifications real\n");
    printf("@ATTRIBUTE Error real\n");
    printf("@DATA\n");

    // Print the initial errors
    x1.assign(testFeatures);
    y1.assign(testLabels);
    printf("%5d, %8.2f, %5.0f, %8.4f\n",
        0,
        0.0,
        T(miss.evaluate(true)),
        T(error.evaluate(true)));
    cout.flush();

    Rand rand(42);
    const size_t batchSize = 100;
    BatchIterator<T> it(trainFeatures, trainLabels, batchSize, rand);
    Tensor<T>* batchFeatures;
    Tensor<T>* batchLabels;

    Timer t;
    for (size_t i = 0; i < 3; ++i)
    {
        // Perform one epoch using batches of the training set
        while (it.hasNext())
        {
            it.next(batchFeatures, batchLabels);
            x1.assign(*batchFeatures);
            y1.assign(*batchLabels);
            update.evaluate(true);
        }
        it.reset();

        // Use the testing set for evaluation
        x1.assign(testFeatures);
        y1.assign(testLabels);
        printf("%5zu, %8.2f, %5.0f, %8.4f\n",
            (i + 1),
            t.getElapsedTimeSeconds(),
            T(miss.evaluate(true)),
            T(error.evaluate(true)));
        cout.flush();
    }

    Instrumentor::instance().print();

    // vector<Graph<T>> targets({w1, b1});
    // if (validate(error, targets))
    //     cout << "TEST PASSED!" << endl;

    return 0;
}

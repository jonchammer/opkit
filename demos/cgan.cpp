// Comment to disable debug assertions
// #define NDEBUG

#include <iostream>
#include <unordered_map>
#include <set>
#include <vector>

#include "opkit/opkit.h"
#include "Plot.h"

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

template <class T>
Tensor<T> xavierInit(const size_t m, const size_t n, Rand& rand)
{
    T stdev = T{1} / sqrt(m / T{2});
    return normal<T>({m, n}, rand, 0, stdev);
}

template <class T>
Tensor<T> sampleZ(const size_t m, const size_t n, Rand& rand)
{
    return uniform<T>({m, n}, rand, -1, 1);
}

template <class T>
Graph<T> generator(const Graph<T>& z, const Graph<T>& y,
    const Graph<T>& gW1, const Graph<T>& gW1b, const Graph<T>& gB1,
    const Graph<T>& gW2, const Graph<T>& gB2)
{
    auto gH1 = relu(matrixMultiply(z, gW1) + matrixMultiply(y, gW1b) + gB1);
    return logistic(linear(gH1, gW2, gB2));
}

template <class T>
Graph<T> discriminator(const Graph<T>& x, Graph<T>& y,
    const Graph<T>& dW1, const Graph<T>& dW1b, const Graph<T>& dB1,
    const Graph<T>& dW2, const Graph<T>& dB2)
{
    auto dH1 = relu(matrixMultiply(x, dW1) + matrixMultiply(y, dW1b) + dB1);
    return logistic(linear(dH1, dW2, dB2));
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

    // Construct the variables
    const size_t zDim            = 128;
    const size_t batchSize       = 128;
    const size_t samplesPerClass = 5;
    const size_t featureDims     = 784;
    const size_t labelDims       = 10;
    Rand initializer(42);

    // Noise and labels used for plotting
    Tensor<T> testZ = sampleZ<T>(samplesPerClass * labelDims, zDim, initializer);
    Tensor<T> testY = zeroes<T>({samplesPerClass * labelDims, labelDims});
    for (size_t i = 0; i < samplesPerClass * labelDims; ++i)
        testY.at({i, i/samplesPerClass}) = T{1};

    // Create the Graph variables
    auto x    = make_variable<T>("x", trainFeatures);
    auto y    = make_variable<T>("y", trainLabels);
    auto z    = make_variable<T>("z", zeroes<T>({1, zDim}));

    auto dW1  = make_variable<T>("dW1", xavierInit<T>(featureDims, 128, initializer));
    auto dW1b = make_variable<T>("dW1b", xavierInit<T>(labelDims, 128, initializer));
    auto dB1  = make_variable<T>("dB1", zeroes<T>({1, 128}));
    auto dW2  = make_variable<T>("dW2", xavierInit<T>(128, 1, initializer));
    auto dB2  = make_variable<T>("dB2", zeroes<T>({1, 1}));
    std::unordered_set<std::string> dNames({"dW1", "dW1b", "dB1", "dW2", "dB2"});
    std::vector<Graph<T>>           dVars({dW1, dW1b, dB1, dW2, dB2});

    auto gW1  = make_variable<T>("gW1", xavierInit<T>(zDim, 128, initializer));
    auto gW1b = make_variable<T>("gW1b", xavierInit<T>(labelDims, 128, initializer));
    auto gB1  = make_variable<T>("gB1", zeroes<T>({1, 128}));
    auto gW2  = make_variable<T>("gW2", xavierInit<T>(128, featureDims, initializer));
    auto gB2  = make_variable<T>("gB2", zeroes<T>({1, featureDims}));
    std::unordered_set<std::string> gNames({"gW1", "gW1b", "gB1", "gW2", "gB2"});

    // Build the graph with error functions
    auto gSample = generator(z, y, gW1, gW1b, gB1, gW2, gB2);
    auto dReal   = discriminator(      x, y, dW1, dW1b, dB1, dW2, dB2);
    auto dFake   = discriminator(gSample, y, dW1, dW1b, dB1, dW2, dB2);

    // Build the loss functions and the optimizer
    auto dLoss = -reduceMean(log(dReal) + log(T{1} - dFake));
    auto gLoss = -reduceMean(log(dFake));

    auto dSolver = adam(dLoss, dNames, T{1E-3});
    auto gSolver = adam(gLoss, gNames, T{1E-3});

    Rand rand(42);
    BatchIterator<T> it(trainFeatures, trainLabels, batchSize, rand);
    Tensor<T>* batchFeatures;
    Tensor<T>* batchLabels;

    printf("%5s, %8s, %8s, %8s\n", "It", "Time", "gLoss", "dLoss");

    Timer t;
    size_t j = 0;
    for (size_t i = 0; i < 100000; ++i)
    {
        it.next(batchFeatures, batchLabels);

        x.assign(*batchFeatures);
        y.assign(*batchLabels);
        z.assign(sampleZ<T>(batchSize, zDim, initializer));
        dSolver.evaluate(true);

        z.assign(sampleZ<T>(batchSize, zDim, initializer));
        gSolver.evaluate(true);

        if (i % 1000 == 0)
        {
            // Print the loss values to demonstrate that everything's working
            printf("%5zu, %8.2f, %8.4f, %8.4f\n",
                j,
                t.getElapsedTimeSeconds(),
                T(gLoss.evaluate(true)),
                T(dLoss.evaluate(true)));
            cout.flush();

            // Save a few samples to demonstrate the system is working
            z.assign(testZ);
            y.assign(testY);
            Tensor<T> samples = gSample.evaluate(true);
            string filename = "./out/img_" + to_string(j) + ".png";
            if (!plotGrid(filename, samples, labelDims, samplesPerClass, 28, 28, 2, 2, 2, 2))
                cerr << "Unable to create file: " << filename << endl;
            ++j;
        }

        if (!it.hasNext())
            it.reset();
    }

    return 0;
}

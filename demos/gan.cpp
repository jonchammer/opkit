// Comment to disable debug assertions
#define NDEBUG

#include <iostream>
#include <unordered_map>
#include <set>
#include <vector>

#include "opkit/opkit.h"
#include "Plot.h"

using namespace std;
using namespace opkit;

template <class T>
Tensor<T> sampleZ(const size_t m, const size_t n, Rand& rand)
{
    return uniform<T>({m, n}, rand, -1, 1);
}

template <class T>
Graph<T> generator(Graph<T> z,
     Graph<T> gW1, Graph<T> gB1,
     Graph<T> gW2, Graph<T> gB2)
{
    auto gH1 = relu(linear(z, gW1, gB1));
    return logistic(linear(gH1, gW2, gB2));
}

template <class T>
Graph<T> discriminator(Graph<T> x,
    Graph<T> dW1, Graph<T> dB1,
    Graph<T> dW2, Graph<T> dB2)
{
    auto dH1 = relu(linear(x, dW1, dB1));
    return logistic(linear(dH1, dW2, dB2));
}

int main()
{
    using T = float;

    // Load the training data
    Tensor<T> trainFeatures, trainLabels;
    loadTensorRaw("/home/jhammer/data/mnist/mnist_train_features_float.raw", trainFeatures);
    loadTensorRaw("/home/jhammer/data/mnist/mnist_train_labels_float.raw",   trainLabels);

    // Perform preprocessing
    scale(trainFeatures, 1.0 / 255.0);
    trainLabels = convertColumnToOneHot(trainLabels, 0);

    // Construct the variables
    const size_t zDim            = 100;
    const size_t batchSize       = 128;
    const size_t plotSamples     = 16;
    const size_t featureDims     = 784;
    Rand initializer(42);

    // Noise used for plotting
    Tensor<T> testZ = sampleZ<T>(plotSamples, zDim, initializer);

    // Create the Graph variables
    auto x    = make_variable<T>("x", trainFeatures);
    auto z    = make_variable<T>("z", zeroes<T>({1, zDim}));

    auto dW1  = make_variable<T>("dW1", xavier<T>({featureDims, 128}, initializer));
    auto dB1  = make_variable<T>("dB1", zeroes<T>({1, 128}));
    auto dW2  = make_variable<T>("dW2", xavier<T>({128, 1}, initializer));
    auto dB2  = make_variable<T>("dB2", zeroes<T>({1, 1}));
    std::unordered_set<std::string> dNames({"dW1", "dB1", "dW2", "dB2"});
    std::vector<Graph<T>>           dVars({dW1, dB1, dW2, dB2});

    auto gW1  = make_variable<T>("gW1", xavier<T>({zDim, 128}, initializer));
    auto gB1  = make_variable<T>("gB1", zeroes<T>({1, 128}));
    auto gW2  = make_variable<T>("gW2", xavier<T>({128, featureDims}, initializer));
    auto gB2  = make_variable<T>("gB2", zeroes<T>({1, featureDims}));
    std::unordered_set<std::string> gNames({"gW1", "gB1", "gW2", "gB2"});

    // Build the graph with error functions
    auto gSample = generator(          z, gW1, gB1, gW2, gB2);
    auto dReal   = discriminator(      x, dW1, dB1, dW2, dB2);
    auto dFake   = discriminator(gSample, dW1, dB1, dW2, dB2);

    // Build the loss functions and the optimizer
    auto dLoss = -reduceMean(log(dReal) + log(T{1} - dFake));
    auto gLoss = -reduceMean(log(dFake));

    // Build the update rule
    auto dSolver = adam(dLoss, dNames, 1E-4);
    auto gSolver = adam(gLoss, gNames, 1E-4);

    Rand rand(42);
    BatchIterator<T> it(trainFeatures, trainLabels, batchSize, rand);
    Tensor<T>* batchFeatures;
    Tensor<T>* batchLabels;

    printf("%5s, %8s, %8s, %8s\n", "It", "Time", "gLoss", "dLoss");

    Timer t;
    size_t j = 0;
    for (size_t i = 0; i < 100000; ++i)
    {
        // Get the next batch
        it.next(batchFeatures, batchLabels);
        if (!it.hasNext())
            it.reset();

        // Update the discriminator
        x.assign(*batchFeatures);
        z.assign(sampleZ<T>(batchSize, zDim, initializer));
        dSolver();

        // Update the generator
        z.assign(sampleZ<T>(batchSize, zDim, initializer));
        gSolver();

        if (i % 1000 == 0)
        {
            // Print the loss values to demonstrate that everything's working
            printf("%5zu, %8.2f, %8.4f, %8.4f\n",
                j,
                t.getElapsedTimeSeconds(),
                T(gLoss()),
                T(dLoss()));
            cout.flush();

            // Save a few samples to demonstrate the system is working
            z.assign(testZ);
            Tensor<T> samples = gSample();
            string filename = "./out/img_" + to_string(j) + ".png";
            if (!plotGrid(filename, samples, 4, 4, 28, 28, 2, 2, 2, 2))
                cerr << "Unable to create file: " << filename << endl;
            ++j;
        }
    }

    return 0;
}

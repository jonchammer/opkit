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
Tensor<T> sampleZ(const size_t m, const size_t n, Rand& rand)
{
    return uniform<T>({m, n}, rand, -1, 1);
}

template <class T>
Graph<T> generator(const Graph<T>& z,
    const Graph<T>& gW1, const Graph<T>& gB1,
    const Graph<T>& gW2, const Graph<T>& gB2)
{
    auto gH1 = relu(linear(z, gW1, gB1));
    return logistic(linear(gH1, gW2, gB2));
}

template <class T>
Graph<T> discriminator(const Graph<T>& x,
    const Graph<T>& dW1, const Graph<T>& dB1,
    const Graph<T>& dW2, const Graph<T>& dB2)
{
    auto dH1 = relu(linear(x, dW1, dB1));
    return linear(dH1, dW2, dB2);
}

template <class T, class U>
Graph<T> clipWeights(const vector<Graph<T>>& nodes, const U min, const U max)
{
    vector<Graph<T>> rules;
    for (const Graph<T>& elem : nodes)
        rules.emplace_back(assign(elem, clip(elem, min, max)));
    return list(rules);
}

int main()
{
    using T = float;

    // Load the training data
    Tensor<T> trainFeatures, trainLabels, testFeatures, testLabels;
    loadTensorRaw("/home/jhammer/data/mnist/mnist_train_features_float.raw", trainFeatures);
    loadTensorRaw("/home/jhammer/data/mnist/mnist_train_labels_float.raw",   trainLabels);

    // Perform preprocessing
    scale(trainFeatures, T{1.0 / 255.0});
    trainLabels = convertColumnToOneHot(trainLabels, 0);

    // Construct the variables
    const size_t xDim        = 784;
    const size_t hDim        = 128;
    const size_t zDim        = 10;
    const size_t batchSize   = 32;
    const size_t plotSamples = 16;
    Rand initializer(53);

    // Noise used for plotting
    Tensor<T> testZ = sampleZ<T>(plotSamples, zDim, initializer);

    // Create the Graph variables
    auto x    = make_variable<T>("x", sub(trainFeatures, {{0, 1}}));
    auto z    = make_variable<T>("z", zeroes<T>({2, zDim}));

    auto dW1  = make_variable<T>("dW1", xavier<T>({xDim, hDim}, initializer));
    auto dB1  = make_variable<T>("dB1", zeroes<T>({1, hDim}));
    auto dW2  = make_variable<T>("dW2", xavier<T>({hDim, 1}, initializer));
    auto dB2  = make_variable<T>("dB2", zeroes<T>({1, 1}));
    std::unordered_set<std::string> dNames({"dW1", "dB1", "dW2", "dB2"});
    std::vector<Graph<T>>           dVars({dW1, dB1, dW2, dB2});

    auto gW1  = make_variable<T>("gW1", xavier<T>({zDim, hDim}, initializer));
    auto gB1  = make_variable<T>("gB1", zeroes<T>({1, hDim}));
    auto gW2  = make_variable<T>("gW2", xavier<T>({hDim, xDim}, initializer));
    auto gB2  = make_variable<T>("gB2", zeroes<T>({1, xDim}));
    std::unordered_set<std::string> gNames({"gW1", "gB1", "gW2", "gB2"});
    std::vector<Graph<T>> gVars({gW1, gB1, gW2, gB2});

    // Build the graph with error functions
    auto gSample = generator(          z, gW1, gB1, gW2, gB2);
    auto dReal   = discriminator(      x, dW1, dB1, dW2, dB2);
    auto dFake   = discriminator(gSample, dW1, dB1, dW2, dB2);

    // WGAN Loss
    auto dLoss = reduceMean(dFake) - reduceMean(dReal); // minimizing -(original loss)
    auto gLoss = -reduceMean(dFake);
    auto clipD = clipWeights(dVars, -0.01, 0.01);

    // Build the update rule
    auto dSolver = rmsProp(dLoss, dNames, 5E-4, 0.9, 0.0, 1E-10);
    auto gSolver = rmsProp(gLoss, gNames, 5E-4, 0.9, 0.0, 1E-10);

    Rand rand(42);
    BatchIterator<T> it(trainFeatures, trainLabels, batchSize, rand);
    Tensor<T>* batchFeatures;
    Tensor<T>* batchLabels;

    printf("%5s, %8s, %8s, %8s\n", "It", "Time", "gLoss", "dLoss");
    Timer t;
    size_t j = 0;
    for (size_t i = 0; i < 1000000; ++i)
    {
        for (size_t k = 0; k < 5; ++k)
        {
            // Get the next batch
            it.next(batchFeatures, batchLabels);
            if (!it.hasNext())
                it.reset();

            // Update the discriminator
            x.assign(*batchFeatures);
            z.assign(sampleZ<T>(batchSize, zDim, initializer));
            dSolver();

            // Clip the weights of the discriminator
            clipD();
        }

        // Update the generator
        z.assign(sampleZ<T>(batchSize, zDim, initializer));
        gSolver();

        if (i % 100 == 0)
        {
            // Print the loss values to demonstrate that everything's working
            printf("%5zu, %8.2f, %8.4f, %8.4f\n",
                i,
                t.getElapsedTimeSeconds(),
                T(gLoss()),
                T(dLoss()));
            cout.flush();

            if (i % 1000 == 0)
            {
                // Save a few samples to demonstrate the system is working
                z.assign(testZ);
                Tensor<T> samples = gSample();
                string filename = "./out/img_" + to_string(j) + ".png";
                if (!plotGrid(filename, samples, 4, 4, 28, 28, 2, 2, 2, 2))
                    cerr << "Unable to create file: " << filename << endl;
                ++j;
            }
        }
    }

    return 0;
}

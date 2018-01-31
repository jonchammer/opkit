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
Graph<T> reshape(Graph<T> x, Graph<T> newShape)
{
    registerDerivative<T>("reshape",
        [](Graph<T> node, Graph<T> delta, std::vector<Graph<T>>& gradients)
    {
        gradients.push_back(reshape(delta, shape(node)));
    });

    return make_binary("reshape", [](const Tensor<T>& x, const Tensor<T>& shape)
    {
        return reshape(x, shape.begin(), shape.end());
    }, x, newShape);
}

template <class T>
Graph<T> reshape(Graph<T> x, std::initializer_list<size_t> shape)
{
    auto var = make_constant<T>("reshape_shape", Tensor<T>::fromVector(shape));
    return reshape(x, var);
}

template <class T>
Tensor<T> sampleZ(const size_t m, const size_t n, Rand& rand)
{
    return uniform<T>({m, n}, rand, -1, 1);
}

// z:    [batch size x zdim]
// y:    [batch size x ydim]
// att:  [xdim x 2]
// gW1:  [zdim x I]
// gW1b: [ydim x I]
// gW1c: [2 x I]
// gB1:  [I]
// gW2:  [I x num channels]
// gB2:  [num channels]
// out:  [batch size x xdim x num channels]
template <class T>
Graph<T> generator(Graph<T> z, Graph<T> y, Graph<T> att,
    Graph<T> gW1, Graph<T> gW1b, Graph<T> gW1c, Graph<T> gB1,
    Graph<T> gW2, Graph<T> gB2)
{
    auto temp = matrixMultiply(z, gW1) + matrixMultiply(y, gW1b);
    auto gH1  = relu(reshape(temp, {b, 1, 128}) +
        reshape(matrixMultiply(att, gW1c), {1, N, 128}) + gB1);
    return logistic(linear(gH1, gW2, gB2));
}



template <class T>
Graph<T> discriminator(Graph<T> x, Graph<T> y,
    Graph<T> dW1, Graph<T> dW1b, Graph<T> dB1,
    Graph<T> dW2, Graph<T> dB2)
{
    auto dH1 = relu(matrixMultiply(x, dW1) + matrixMultiply(y, dW1b) + dB1);
    return logistic(linear(dH1, dW2, dB2));
}

int main()
{
    // TODO:
    // Update matrixMultiply to work with batches. [b x M x N] * [N * P] = [b x M x P]
    // Verify attVals has the correct numbers.
    // Make reshape ops in generator() syntactically correct.
    // Get framework to compile correctly outside of the package.
    
    using T = float;

    // Load the training data
    Tensor<T> trainFeatures, trainLabels;
    loadTensorRaw("/home/jhammer/data/mnist/mnist_train_features_float.raw", trainFeatures);
    loadTensorRaw("/home/jhammer/data/mnist/mnist_train_labels_float.raw",   trainLabels);

    // Perform preprocessing
    scale(trainFeatures, T{1.0 / 255.0});
    trainLabels = convertColumnToOneHot(trainLabels, 0);

    // Construct the variables
    const size_t zDim            = 100;
    const size_t batchSize       = 1; // orig = 128;
    const size_t samplesPerClass = 5;
    const size_t featureDims     = 784;
    const size_t labelDims       = 10;
    const size_t attDims         = 2;
    const size_t rgbDims         = 1; // Or 3 for actual RGB
    Rand initializer(42);

    // Noise and labels used for plotting
    Tensor<T> testZ = sampleZ<T>(samplesPerClass * labelDims, zDim, initializer);
    Tensor<T> testY = zeroes<T>({samplesPerClass * labelDims, labelDims});
    for (size_t i = 0; i < samplesPerClass * labelDims; ++i)
        testY.at({i, i/samplesPerClass}) = T{1};

    // The attention values divide the image into a 28 x 28, scaled between 0 and 1.
    Tensor<T> attVals({featureDims, attDims});
    for (size_t i = 0; i < featureDims; ++i)
    {
        attVals.at({i, 0}) = (i / 28) / 28.0;
        attVals.at({i, 1}) = (i % 28) / 28.0;
    }

    // Create the Graph variables
    auto x    = make_variable<T>("x",   trainFeatures);
    auto y    = make_variable<T>("y",   trainLabels);
    auto z    = make_variable<T>("z",   zeroes<T>({1, zDim}));
    auto att  = make_variable<T>("att", attVals);

    auto dW1  = make_variable<T>("dW1",  xavier<T>({featureDims, 128}, initializer));
    auto dW1b = make_variable<T>("dW1b", xavier<T>({labelDims, 128}, initializer));
    auto dB1  = make_variable<T>("dB1",  zeroes<T>({1, 128}));
    auto dW2  = make_variable<T>("dW2",  xavier<T>({128, 1}, initializer));
    auto dB2  = make_variable<T>("dB2",  zeroes<T>({1, 1}));
    std::unordered_set<std::string> dNames({"dW1", "dW1b", "dB1", "dW2", "dB2"});
    std::vector<Graph<T>>           dVars({dW1, dW1b, dB1, dW2, dB2});

    auto gW1  = make_variable<T>("gW1",  xavier<T>({zDim, 128}, initializer));
    auto gW1b = make_variable<T>("gW1b", xavier<T>({labelDims, 128}, initializer));
    auto gW1c = make_variable<T>("gW1c", xavier<T>({attDims, 128}, initializer));
    auto gB1  = make_variable<T>("gB1",  zeroes<T>({1, 128}));
    auto gW2  = make_variable<T>("gW2",  xavier<T>({128, rgbDims}, initializer));
    auto gB2  = make_variable<T>("gB2",  zeroes<T>({1, rgbDims}));
    std::unordered_set<std::string> gNames({"gW1", "gW1b", "gW1c", "gB1", "gW2", "gB2"});

    // Build the graph with error functions
    auto gSample = generator    (z, y, att, gW1, gW1b, gW1c, gB1, gW2, gB2);
    auto dReal   = discriminator(      x, y, dW1, dW1b, dB1, dW2, dB2);
    auto dFake   = discriminator(reshape(gSample, {batchSize, featureDims * rgbDims}), y, dW1, dW1b, dB1, dW2, dB2);

    // Build the loss functions and the optimizer
    auto dLoss = -reduceMean(log(dReal) + log(T{1} - dFake));
    auto gLoss = -reduceMean(log(dFake));

    // auto dSolver = gradientDescentMomentum(dLoss, dNames);
    // auto gSolver = gradientDescentMomentum(gLoss, gNames);
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
        if (!it.hasNext())
            it.reset();

        x.assign(*batchFeatures);
        y.assign(*batchLabels);
        z.assign(sampleZ<T>(batchSize, zDim, initializer));
        dSolver();

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
            y.assign(testY);
            Tensor<T> samples = gSample();
            string filename = "./out/img_" + to_string(j) + ".png";
            if (!plotGrid(filename, samples, labelDims, samplesPerClass,
                28, 28, 2, 2, 2, 2))
                cerr << "Unable to create file: " << filename << endl;
            ++j;
        }
    }

    return 0;
}

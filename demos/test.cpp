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
Tensor<T> fromTxt(const string& filename)
{
    ifstream din(filename);
    if (!din)
        throw std::runtime_error("Unable to open file: " + filename);

    // Work out how many rows and columns there are
    size_t rows = std::count(std::istreambuf_iterator<char>(din),
        std::istreambuf_iterator<char>(), '\n');
    din.seekg(0);
    string line;
    getline(din, line);
    size_t cols = 1 + std::count_if(line.begin(), line.end(), [](unsigned char c)
    {
        return std::isspace(c);
    });
    din.seekg(0);
    // cout << "Rows: " << rows << " Cols: " << cols << endl;

    // Read the data into a Storage object
    Storage<T> storage(rows * cols);
    T val;
    size_t i = 0;
    while (din >> val)
        storage[i++] = val;

    din.close();
    return Tensor<T>(storage, {rows, cols});
}

template <class T>
bool operator==(const Tensor<T>& a, const Tensor<T>& b)
{
    // cout << to_string(a, 4) << endl << "vs" << endl << to_string(b, 4) << endl << endl;
    cout << to_string(reduceMean(sub(a, b)), 4) << endl;
    if (a.shape() == b.shape())
    {
        auto aIt = a.begin();
        auto bIt = b.begin();
        while (aIt != a.end())
        {
            if (abs(*aIt - *bIt) > 1E-4)
                return false;
            ++aIt;
            ++bIt;
        }

        return true;
    }
    else return false;
}

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
    return make_component("generator", logistic(linear(gH1, gW2, gB2)));
}

template <class T>
Graph<T> discriminator(const string& name, const Graph<T>& x,
    const Graph<T>& dW1, const Graph<T>& dB1,
    const Graph<T>& dW2, const Graph<T>& dB2)
{
    auto dH1 = relu(linear(x, dW1, dB1));
    return make_component(name, linear(dH1, dW2, dB2));
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

    Tensor<T> dw1Vals  = fromTxt<T>("/home/jhammer/projects/Tensorflow-Tests/gans/dw1.txt");
    Tensor<T> dw12Vals = fromTxt<T>("/home/jhammer/projects/Tensorflow-Tests/gans/dw12.txt");
    Tensor<T> dw2Vals  = fromTxt<T>("/home/jhammer/projects/Tensorflow-Tests/gans/dw2.txt");
    Tensor<T> dw22Vals = fromTxt<T>("/home/jhammer/projects/Tensorflow-Tests/gans/dw22.txt");
    Tensor<T> db12Vals = fromTxt<T>("/home/jhammer/projects/Tensorflow-Tests/gans/db12.txt");
    Tensor<T> db22Vals = fromTxt<T>("/home/jhammer/projects/Tensorflow-Tests/gans/db22.txt");
    db12Vals.resize({1, 128});

    Tensor<T> gw1Vals  = fromTxt<T>("/home/jhammer/projects/Tensorflow-Tests/gans/gw1.txt");
    Tensor<T> gw12Vals = fromTxt<T>("/home/jhammer/projects/Tensorflow-Tests/gans/gw12.txt");
    Tensor<T> gw2Vals  = fromTxt<T>("/home/jhammer/projects/Tensorflow-Tests/gans/gw2.txt");
    Tensor<T> gw22Vals = fromTxt<T>("/home/jhammer/projects/Tensorflow-Tests/gans/gw22.txt");
    Tensor<T> gb12Vals = fromTxt<T>("/home/jhammer/projects/Tensorflow-Tests/gans/gb12.txt");
    Tensor<T> gb22Vals = fromTxt<T>("/home/jhammer/projects/Tensorflow-Tests/gans/gb22.txt");
    gb12Vals.resize({1, 128});
    gb22Vals.resize({1, 784});

    Tensor<T> xmbVals  = fromTxt<T>("/home/jhammer/projects/Tensorflow-Tests/gans/xmb.txt");
    Tensor<T> ZVals    = fromTxt<T>("/home/jhammer/projects/Tensorflow-Tests/gans/testZ.txt");
    Tensor<T> Z2Vals   = fromTxt<T>("/home/jhammer/projects/Tensorflow-Tests/gans/testZ2.txt");

    // Construct the variables
    const size_t xDim        = 784;
    const size_t hDim        = 128;
    const size_t zDim        = 10;
    const size_t batchSize   = 32;
    const size_t plotSamples = 16;
    Rand initializer(42);

    // Noise used for plotting
    Tensor<T> testZ = sampleZ<T>(plotSamples, zDim, initializer);

    // Create the Graph variables
    auto x    = make_variable<T>("x", xmbVals);
    auto z    = make_variable<T>("z", ZVals);

    auto dW1  = make_variable<T>("dW1", dw1Vals);
    auto dB1  = make_variable<T>("dB1", zeroes<T>({1, hDim}));
    auto dW2  = make_variable<T>("dW2", dw2Vals);
    auto dB2  = make_variable<T>("dB2", zeroes<T>({1, 1}));
    std::unordered_set<std::string> dNames({"dW1", "dB1", "dW2", "dB2"});
    std::vector<Graph<T>>           dVars({dW1, dB1, dW2, dB2});

    auto gW1  = make_variable<T>("gW1", gw1Vals);
    auto gB1  = make_variable<T>("gB1", zeroes<T>({1, hDim}));
    auto gW2  = make_variable<T>("gW2", gw2Vals);
    auto gB2  = make_variable<T>("gB2", zeroes<T>({1, xDim}));
    std::unordered_set<std::string> gNames({"gW1", "gB1", "gW2", "gB2"});
    std::vector<Graph<T>> gVars({gW1, gB1, gW2, gB2});

    // Build the graph with error functions
    auto gSample = generator(                   z, gW1, gB1, gW2, gB2);
    auto dReal   = discriminator("dReal",       x, dW1, dB1, dW2, dB2);
    auto dFake   = discriminator("dFake", gSample, dW1, dB1, dW2, dB2);

    // WGAN Loss
    // minimizing -(original loss)
    auto dLoss = make_component("dLoss", reduceMean(dFake) - reduceMean(dReal));
    auto gLoss = make_component("gLoss", -reduceMean(dFake));
    auto clipD = clipWeights(dVars, -0.01, 0.01);

    // std::vector<Graph<T>> allVars({gW1, gB1, gW2, gB2, dW1, dB1, dW2, dB2, x, z});
    // if (validate(gLoss, allVars, 1E-1))
    // {
    //     cout << "PASSED" << endl;
    // }
    // else cout << "FAILED" << endl;

    // Build the update rule
    auto dSolver = rmsProp(dLoss, dNames, 1E-4);
    auto gSolver = rmsProp(gLoss, gNames, 1E-4);

    auto dGrads = gradients(dLoss, dNames);
    for (auto& pair : dGrads)
    {
        cout << pair.first << " -> " << pair.second << endl;
    }
    // Update the discriminator
    dSolver();
    clipD();

    cout << "DW1 " << (dW1() == dw12Vals ? "Passed" : "Failed") << endl;
    cout << "DB1 " << (dB1() == db12Vals ? "Passed" : "Failed") << endl;
    cout << "DW2 " << (dW2() == dw22Vals ? "Passed" : "Failed") << endl;
    cout << "DB2 " << (dB2() == db22Vals ? "Passed" : "Failed") << endl;

    // Update the generator
    z.assign(Z2Vals);
    gSolver();

    cout << "GW1 " << (gW1() == gw12Vals ? "Passed" : "Failed") << endl;
    cout << "GB1 " << (gB1() == gb12Vals ? "Passed" : "Failed") << endl;
    cout << "GW2 " << (gW2() == gw22Vals ? "Passed" : "Failed") << endl;
    cout << "GB2 " << (gB2() == gb22Vals ? "Passed" : "Failed") << endl;

    Timer t;
    printf("%5s, %8s, %8s, %8s\n", "It", "Time", "gLoss", "dLoss");
    printf("%5zu, %8.2f, %8.4f, %8.4f\n",
        0ul,
        t.getElapsedTimeSeconds(),
        T(gLoss()),
        T(dLoss()));
    cout.flush();

    return 0;
}

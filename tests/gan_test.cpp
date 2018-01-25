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
    if (a.shape() == b.shape())
    {
        auto aIt = a.begin();
        auto bIt = b.begin();
        while (aIt != a.end())
        {
            if (abs(*aIt - *bIt) > 1)
            {
                cout << "A: " << endl << to_string(a, 8) << endl;
                cout << "B: " << endl << to_string(b, 8) << endl;
                cout << to_string(reduceMean(sub(a, b)), 8) << endl;
                return false;
            }
            ++aIt;
            ++bIt;
        }

        return true;
    }
    else
    {
        cout << "Shape Mismatch: " << a.shape() << " vs. " << b.shape() << endl;
        return false;
    }
}

//----------------------------------------------------------------------------//

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

    string dir = "../gan_test_files/";
    Tensor<T> dw1Vals  = fromTxt<T>(dir + "dw1.txt");
    Tensor<T> dw12Vals = fromTxt<T>(dir + "dw12.txt");
    Tensor<T> dw2Vals  = fromTxt<T>(dir + "dw2.txt");
    Tensor<T> dw22Vals = fromTxt<T>(dir + "dw22.txt");
    Tensor<T> db12Vals = fromTxt<T>(dir + "db12.txt");
    Tensor<T> db22Vals = fromTxt<T>(dir + "db22.txt");
    db12Vals.resize({1, 128});

    Tensor<T> gw1Vals  = fromTxt<T>(dir + "gw1.txt");
    Tensor<T> gw12Vals = fromTxt<T>(dir + "gw12.txt");
    Tensor<T> gw2Vals  = fromTxt<T>(dir + "gw2.txt");
    Tensor<T> gw22Vals = fromTxt<T>(dir + "gw22.txt");
    Tensor<T> gb12Vals = fromTxt<T>(dir + "gb12.txt");
    Tensor<T> gb22Vals = fromTxt<T>(dir + "gb22.txt");
    gb12Vals.resize({1, 128});
    gb22Vals.resize({1, 784});

    Tensor<T> xmbVals  = fromTxt<T>(dir + "xmb.txt");
    Tensor<T> ZVals    = fromTxt<T>(dir + "testZ.txt");
    Tensor<T> Z2Vals   = fromTxt<T>(dir + "testZ2.txt");

    Tensor<T> gradDw1Vals = fromTxt<T>(dir + "grad_dw1.txt");
    Tensor<T> gradDb1Vals = fromTxt<T>(dir + "grad_db1.txt");
    Tensor<T> gradDw2Vals = fromTxt<T>(dir + "grad_dw2.txt");
    Tensor<T> gradDb2Vals = fromTxt<T>(dir + "grad_db2.txt");
    Tensor<T> gradGw1Vals = fromTxt<T>(dir + "grad_gw1.txt");
    Tensor<T> gradGb1Vals = fromTxt<T>(dir + "grad_gb1.txt");
    Tensor<T> gradGw2Vals = fromTxt<T>(dir + "grad_gw2.txt");
    Tensor<T> gradGb2Vals = fromTxt<T>(dir + "grad_gb2.txt");
    gradDb1Vals.resize({1, 128});
    gradDb2Vals.resize({1, 1});
    gradGb1Vals.resize({1, 128});
    gradGb2Vals.resize({1, 784});

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

    // Build the update rule
    auto dSolver = rmsProp(dLoss, dNames, 1E-4, 0.9, 0.0, 1E-10);
    auto gSolver = rmsProp(gLoss, gNames, 1E-4, 0.9, 0.0, 1E-10);

    // Check the gradients
    auto dGrads = gradients(dLoss, dNames);
    cout << "1.  Grad DW1 " << (dGrads["dW1"]() == gradDw1Vals ? "Passed" : "Failed") << endl;
    cout << "2.  Grad DB1 " << (dGrads["dB1"]() == gradDb1Vals ? "Passed" : "Failed") << endl;
    cout << "3.  Grad DW2 " << (dGrads["dW2"]() == gradDw2Vals ? "Passed" : "Failed") << endl;
    cout << "4.  Grad DB2 " << (dGrads["dB2"]() == gradDb2Vals ? "Passed" : "Failed") << endl;

    auto gGrads = gradients(gLoss, gNames);
    cout << "5.  Grad GW1 " << (gGrads["gW1"]() == gradGw1Vals ? "Passed" : "Failed") << endl;
    cout << "6.  Grad GB1 " << (gGrads["gB1"]() == gradGb1Vals ? "Passed" : "Failed") << endl;
    cout << "7.  Grad GW2 " << (gGrads["gW2"]() == gradGw2Vals ? "Passed" : "Failed") << endl;
    cout << "8.  Grad GB2 " << (gGrads["gB2"]() == gradGb2Vals ? "Passed" : "Failed") << endl;

    // Update the discriminator
    dSolver();
    clipD();

    cout << "9.  DW1 " << (dW1() == dw12Vals ? "Passed" : "Failed") << endl;
    cout << "10. DB1 " << (dB1() == db12Vals ? "Passed" : "Failed") << endl;
    cout << "11. DW2 " << (dW2() == dw22Vals ? "Passed" : "Failed") << endl;
    cout << "12. DB2 " << (dB2() == db22Vals ? "Passed" : "Failed") << endl;

    // Update the generator
    z.assign(Z2Vals);
    gSolver();

    cout << "13. GW1 " << (gW1() == gw12Vals ? "Passed" : "Failed") << endl;
    cout << "14. GB1 " << (gB1() == gb12Vals ? "Passed" : "Failed") << endl;
    cout << "15. GW2 " << (gW2() == gw22Vals ? "Passed" : "Failed") << endl;
    cout << "16. GB2 " << (gB2() == gb22Vals ? "Passed" : "Failed") << endl;

    cout << "17. GLoss " << (abs(T(gLoss()) -  0.018546745) < 0.01 ? "Passed" : "Failed") << endl;
    cout << "18. DLoss " << (abs(T(dLoss()) - -0.015703537) < 0.01 ? "Passed" : "Failed") << endl;

    Timer t;
    printf("%5s, %8s, %8s, %8s\n", "It", "Time", "gLoss", "dLoss");
    printf("%5zu, %8.2f, %8.8f, %8.8f\n",
        0ul,
        t.getElapsedTimeSeconds(),
        T(gLoss()),
        T(dLoss()));
    cout.flush();

    return 0;
}

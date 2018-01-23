#include <iostream>

#include "opkit/opkit.h"

using namespace std;
using namespace opkit;

using T = float;

// Keep track of how many times the test bench is called
int callCounter = 0;

template <class Fn>
void testbench(const string& name, const T* xTruth, const T* yTruth, Fn&& optimizer)
{
    // Build the test graph
    auto x         = make_variable<T>("x", Tensor<T>::fromScalar(5.0));
    auto y         = 2.0 * square(x) - 4.0;
    auto trainStep = optimizer(y, {"x"});

    // cout << trainStep << endl << endl;
    // Run through the training steps
    static const int EPOCHS = 10;
    T xReal[EPOCHS];
    T yReal[EPOCHS];

    for (int i = 0; i < EPOCHS; ++i)
    {
        // cout << ((Variable<T>&) x.node()).value() << endl;
        // cout << ((InPlaceBinaryFunction<T>&) y.node()).cachedResult() << endl;
        // cout << T(x()) << endl;
        // cout << T(y()) << endl << endl;

        trainStep();

        // cout << ((Variable<T>&) x.node()).value() << endl;
        // cout << ((InPlaceBinaryFunction<T>&) y.node()).cachedResult() << endl;
        xReal[i] = T(x());
        yReal[i] = T(y());

        // cout << xReal[i] << endl;
        // cout << yReal[i] << endl;
    }

    // Print the results
    printf("%3d.", callCounter++);
    for (int i = 0; i < EPOCHS; ++i)
    {
        if (abs(xReal[i] - xTruth[i]) > 0.001 ||
            abs(yReal[i] - yTruth[i]) > 0.001)
        {
            printf("%30s - [ ]\n\n", name.c_str());
            printf("Error at iteration: %d\n", i);
            printf("%25s  - %25s\n", "Expected", "Got");
            for (int j = 0; j < EPOCHS; ++j)
            {
                printf("(%11.8f, %11.8f) - (%11.8f, %11.8f)",
                    xTruth[j], yTruth[j], xReal[j], yReal[j]);
                if (i == j)
                    printf(" <---");
                printf("\n");
            }
            printf("\n");
            return;
        }
    }
    printf("%30s - [x]\n", name.c_str());
}

void testGradientDescent()
{
    T xTruth[] = {3.0, 1.8, 1.08, 0.648, 0.3888, 0.23328, 0.139968, 0.083981, 0.050388, 0.030233};
    T yTruth[] = {14.0, 2.48, -1.6672, -3.160192, -3.697669, -3.891161, -3.960818, -3.985894, -3.994922, -3.99817181};

    testbench("GD", xTruth, yTruth,
    [](Graph<T> y, std::unordered_set<string> targets)
    {
        return gradientDescent(y, targets, 0.1);
    });
}

void testGradientDescentMomentum()
{
    T xTruth[] = {3.0, 1.79799998, 1.07759798, 0.64583838, 0.38707128, 0.231984, 0.13903531, 0.08332824, 0.04994124, 0.02993136};
    T yTruth[] = {14.0, 2.46560764, -1.6775651, -3.16578555, -3.70035172, -3.89236689, -3.96133828, -3.98611283, -3.99501181, -3.99820828};

    testbench("GD Momentum", xTruth, yTruth,
    [](Graph<T> y, std::unordered_set<string> targets)
    {
        return gradientDescentMomentum(y, targets, 0.1, 0.001, false);
    });
}

void testGradientDescentNesterovMomentum()
{
    T xTruth[] = {2.99799991, 1.79759872, 1.0778389, 0.64627147, 0.38750392, 0.2323471, 0.13931516, 0.08353327, 0.05008649, 0.03003183};
    T yTruth[] = {13.97600746, 2.4627223, -1.67652655, -3.16466641, -3.69968152, -3.89202976, -3.96118259, -3.98604441, -3.99498272, -3.99819613};

    testbench("GD Nesterov Momentum", xTruth, yTruth,
    [](Graph<T> y, std::unordered_set<string> targets)
    {
        return gradientDescentMomentum(y, targets, 0.1, 0.001, true);
    });
}

void testAdam()
{
    T xTruth[] = {4.9000001, 4.80005836, 4.70021439, 4.60050917, 4.50098467, 4.40168428, 4.30265093, 4.2039299, 4.10556602, 4.0076046};
    T yTruth[] = {44.02000046, 42.08111954, 40.18403244, 38.32936859, 36.5177269, 34.74964905, 33.02561188, 31.34605408, 29.71134567, 28.12178802};

    testbench("Adam", xTruth, yTruth,
    [](Graph<T> y, std::unordered_set<string> targets)
    {
        return adam(y, targets, 0.1, 0.9, 0.999, 1E-8);
    });
}

int main()
{
    testGradientDescent();
    testGradientDescentMomentum();
    testGradientDescentNesterovMomentum();
    testAdam();

    return 0;
}

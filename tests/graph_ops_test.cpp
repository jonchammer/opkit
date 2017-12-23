#include <iostream>

#include "tensor/Tensor.h"
#include "tensor/TensorMath.h"
#include "tensor/TensorIO.h"

#include "graph/Graph.h"
#include "graph/GradientValidator.h"
#include "graph/ops/GraphOps_all.h"

#include "graph/NN.h"
#include "graph/CostFunctions.h"
#include "graph/Optimizers.h"

using namespace std;
using namespace tensorlib;

using T = float;

template <class T>
bool operator==(const Tensor<T>& a, const Tensor<T>& b)
{
    if (a.size() != b.size()) return false;

    auto aIt = a.begin();
    auto bIt = b.begin();
    while (aIt != a.end())
    {
        if (abs(*aIt - *bIt) > 0.01) return false;
        ++aIt;
        ++bIt;
    }
    return true;
}

void doValidate(Graph<T>& g, vector<Graph<T>>& targets,
    const Tensor<T>& expectedValue, const string& name)
{
    bool correctGradient = validate(g, targets, 1E-2);

    if (g.evaluate(true) == expectedValue)
         cout << "[x]  ";
    else
    {
        cout << "[ ]  ";
        // cout << "Expected: " << expectedValue << endl;
        // cout << "Got:      " << g.evaluate(true) << endl;
    }

    if (correctGradient)
         cout << "[x]  ";
    else cout << "[ ]  ";

    cout << name << endl;
}

Tensor<T> scalar(T val)
{
    return Tensor<T>::fromScalar(val);
}

Tensor<T> rowVec(T val1, T val2, T val3)
{
    return Tensor<T>::fromVector({val1, val2, val3});
}

Tensor<T> colVec(T val1, T val2)
{
    return Tensor<T>::fromVector({val1, val2});
}

void testUnaryOps()
{
    Tensor<T> x1   (Storage<T>({5, 7, 2, 1, 0, 3}), {2, 3});
    Tensor<T> x2   (Storage<T>({5, -7, -2, 1, 0, 3}), {2, 3});
    Tensor<T> x3   (Storage<T>({0.1, -0.1, 0.9, -0.9, 2.5, -2.5}), {2, 3});
    Tensor<T> x4   (Storage<T>({0.1, -0.1, 0.9, -0.9, 2.5, -2.5}), {2, 3});
    Tensor<T> x5   (Storage<T>({5, -7, -2, 1, 0, 3}), {2, 3});
    Tensor<T> x6   (Storage<T>({5, -7, -2, 1, 0, 3}), {2, 3});

    Tensor<T> res1 (Storage<T>({2.236, 2.645, 1.414, 1.0, 0.0, 1.732}), {2, 3});
    Tensor<T> res2 (Storage<T>({5, 7, 2, 1, 0, 3}), {2, 3});
    Tensor<T> res3 (Storage<T>({1, 0, 1, 0, 3, -2}), {2, 3});
    Tensor<T> res4 (Storage<T>({0, -1, 0, -1, 2, -3}), {2, 3});
    Tensor<T> res5 (Storage<T>({25, 49, 4, 1, 0, 9}), {2, 3});
    Tensor<T> res6 (Storage<T>({-5, 7, 2, -1, 0, -3}), {2, 3});

    auto a1 = make_variable<T>("a1", x1);
    auto a2 = make_variable<T>("a2", x2);
    auto a3 = make_variable<T>("a3", x3);
    auto a4 = make_variable<T>("a4", x4);
    auto a5 = make_variable<T>("a5", x5);
    auto a6 = make_variable<T>("a6", x6);

    auto y1 = sqrt(a1);
    auto y2 = abs(a2);
    auto y3 = ceil(a3);
    auto y4 = floor(a4);
    auto y5 = square(a5);
    auto y6 = -a6;

    vector<Graph<T>> targets({a1, a2, a3, a4, a5, a6});
    doValidate(y1, targets, res1, "sqrt()");
    doValidate(y2, targets, res2, "abs()");
    doValidate(y3, targets, res3, "ceil()");
    doValidate(y4, targets, res4, "floor()");
    doValidate(y5, targets, res5, "square()");
    doValidate(y6, targets, res6, "neg()");
}

void testAddition()
{
    Tensor<T> x1  (Storage<T>({5, 7, 2, 1, -1, 3}), {2, 3});
    Tensor<T> x2  (Storage<T>({2, 9, 2, 3, -2, 0}), {2, 3});
    Tensor<T> x3  (Storage<T>({1}), {1});
    Tensor<T> x4  (Storage<T>({3, 1}), {2, 1});
    Tensor<T> x5  (Storage<T>({-1, 2, 3, 8}), {1, 4});

    Tensor<T> res1(Storage<T>({7, 16, 4, 4, -3, 3}), {2, 3});
    Tensor<T> res2(Storage<T>({6, 8, 3, 2, 0, 4}), {2, 3});
    Tensor<T> res3(Storage<T>({2, 5, 6, 11, 0, 3, 4, 9}), {2, 4});
    Tensor<T> res4(Storage<T>({10, 12, 7, 6, 4, 8}), {2, 3});

    auto a1  = make_variable<T>("a1", x1);
    auto a2  = make_variable<T>("a2", x2);
    auto a3  = make_variable<T>("a3", x3);
    auto a4  = make_variable<T>("a4", x4);
    auto a5  = make_variable<T>("a5", x5);

    auto y1 = a1 + a2;
    auto y2 = a1 + a3;
    auto y3 = a3 + a1;
    auto y4 = a4 + a5;
    auto y5 = a5 + a4;
    auto y6 = a1 + T{5};
    auto y7 = T{5} + a1;

    vector<Graph<T>> targets({a1, a2, a3, a4, a5});
    doValidate(y1, targets, res1, "addition() Matrix-Matrix");
    doValidate(y2, targets, res2, "addition() Matrix-Scalar");
    doValidate(y3, targets, res2, "addition() Scalar-Matrix");
    doValidate(y4, targets, res3, "addition() Dual Broadcasting");
    doValidate(y5, targets, res3, "addition() Dual Broadcasting Reverse");
    doValidate(y6, targets, res4, "addition() Matrix-Constant");
    doValidate(y7, targets, res4, "addition() Constant-Matrix");
}

void testSubtraction()
{
    Tensor<T> x1  (Storage<T>({5, 7, 2, 1, -1, 3}), {2, 3});
    Tensor<T> x2  (Storage<T>({2, 9, 2, 3, -2, 0}), {2, 3});
    Tensor<T> x3  (Storage<T>({1}), {1});
    Tensor<T> x4  (Storage<T>({3, 1}), {2, 1});
    Tensor<T> x5  (Storage<T>({-1, 2, 3, 8}), {1, 4});

    Tensor<T> res1(Storage<T>({3, -2, 0, -2, 1, 3}), {2, 3});
    Tensor<T> res2(Storage<T>({4, 6, 1, 0, -2, 2}), {2, 3});
    Tensor<T> res3(Storage<T>({-4, -6, -1, 0, 2, -2}), {2, 3});
    Tensor<T> res4(Storage<T>({4, 1, 0, -5, 2, -1, -2, -7}), {2, 4});
    Tensor<T> res5(Storage<T>({-4, -1, 0, 5, -2, 1, 2, 7}), {2, 4});
    Tensor<T> res6(Storage<T>({0, 2, -3, -4, -6, -2}), {2, 3});
    Tensor<T> res7(Storage<T>({0, -2, 3, 4, 6, 2}), {2, 3});

    auto a1  = make_variable<T>("a1", x1);
    auto a2  = make_variable<T>("a2", x2);
    auto a3  = make_variable<T>("a3", x3);
    auto a4  = make_variable<T>("a4", x4);
    auto a5  = make_variable<T>("a5", x5);

    auto y1 = a1 - a2;
    auto y2 = a1 - a3;
    auto y3 = a3 - a1;
    auto y4 = a4 - a5;
    auto y5 = a5 - a4;
    auto y6 = a1 - T{5};
    auto y7 = T{5} - a1;

    vector<Graph<T>> targets({a1, a2, a3, a4, a5});
    doValidate(y1, targets, res1, "subtraction() Matrix-Matrix");
    doValidate(y2, targets, res2, "subtraction() Matrix-Scalar");
    doValidate(y3, targets, res3, "subtraction() Scalar-Matrix");
    doValidate(y4, targets, res4, "subtraction() Dual Broadcasting");
    doValidate(y5, targets, res5, "subtraction() Dual Broadcasting Reverse");
    doValidate(y6, targets, res6, "subtraction() Matrix-Constant");
    doValidate(y7, targets, res7, "subtraction() Constant-Matrix");
}

void testMultiplication()
{
    Tensor<T> x1  (Storage<T>({5, 7, 2, 1, -1, 3}), {2, 3});
    Tensor<T> x2  (Storage<T>({2, 9, 2, 3, -2, 0}), {2, 3});
    Tensor<T> x3  (Storage<T>({2}), {1});
    Tensor<T> x4  (Storage<T>({3, 1}), {2, 1});
    Tensor<T> x5  (Storage<T>({-1, 2, 3, 8}), {1, 4});

    Tensor<T> res1(Storage<T>({10, 63, 4, 3, 2, 0}), {2, 3});
    Tensor<T> res2(Storage<T>({10, 14, 4, 2, -2, 6}), {2, 3});
    Tensor<T> res3(Storage<T>({10, 14, 4, 2, -2, 6}), {2, 3});
    Tensor<T> res4(Storage<T>({-3, 6, 9, 24, -1, 2, 3, 8}), {2, 4});
    Tensor<T> res5(Storage<T>({-3, 6, 9, 24, -1, 2, 3, 8}), {2, 4});
    Tensor<T> res6(Storage<T>({25, 35, 10, 5, -5, 15}), {2, 3});
    Tensor<T> res7(Storage<T>({25, 35, 10, 5, -5, 15}), {2, 3});

    auto a1  = make_variable<T>("a1", x1);
    auto a2  = make_variable<T>("a2", x2);
    auto a3  = make_variable<T>("a3", x3);
    auto a4  = make_variable<T>("a4", x4);
    auto a5  = make_variable<T>("a5", x5);

    auto y1 = a1 * a2;
    auto y2 = a1 * a3;
    auto y3 = a3 * a1;
    auto y4 = a4 * a5;
    auto y5 = a5 * a4;
    auto y6 = a1 * T{5};
    auto y7 = T{5} * a1;

    vector<Graph<T>> targets({a1, a2, a3, a4, a5});
    doValidate(y1, targets, res1, "multiplication() Matrix-Matrix");
    doValidate(y2, targets, res2, "multiplication() Matrix-Scalar");
    doValidate(y3, targets, res3, "multiplication() Scalar-Matrix");
    doValidate(y4, targets, res4, "multiplication() Dual Broadcasting");
    doValidate(y5, targets, res5, "multiplication() Dual Broadcasting Reverse");
    doValidate(y6, targets, res6, "multiplication() Matrix-Constant");
    doValidate(y7, targets, res7, "multiplication() Constant-Matrix");
}

void testDivision()
{
    Tensor<T> x1  (Storage<T>({5, 7, 2, 1, -1, 3}), {2, 3});
    Tensor<T> x2  (Storage<T>({2, 9, 2, 3, -2, 1}), {2, 3});
    Tensor<T> x3  (Storage<T>({2}), {1});
    Tensor<T> x4  (Storage<T>({3, 1}), {2, 1});
    Tensor<T> x5  (Storage<T>({-1, 2, 3, 8}), {1, 4});

    Tensor<T> res1(Storage<T>({2.5, 0.777, 1.0, 0.333, 0.5, 3.0}), {2, 3});
    Tensor<T> res2(Storage<T>({2.5, 3.5, 1.0, 0.5, -0.5, 1.5}), {2, 3});
    Tensor<T> res3(Storage<T>({0.4, 0.285, 1.0, 2.0, -2.0, 0.666}), {2, 3});
    Tensor<T> res4(Storage<T>({-3.0, 1.5, 1.0, 0.375, -1.0, 0.5, 0.333, 0.125}), {2, 4});
    Tensor<T> res5(Storage<T>({-0.333, 0.666, 1.0, 2.666, -1.0, 2.0, 3.0, 8.0}), {2, 4});
    Tensor<T> res6(Storage<T>({1.0, 1.4, 0.4, 0.2, -0.2, 0.6}), {2, 3});
    Tensor<T> res7(Storage<T>({1.0, 0.714, 2.5, 5.0, -5.0, 1.666}), {2, 3});

    auto a1  = make_variable<T>("a1", x1);
    auto a2  = make_variable<T>("a2", x2);
    auto a3  = make_variable<T>("a3", x3);
    auto a4  = make_variable<T>("a4", x4);
    auto a5  = make_variable<T>("a5", x5);

    auto y1 = a1 / a2;
    auto y2 = a1 / a3;
    auto y3 = a3 / a1;
    auto y4 = a4 / a5;
    auto y5 = a5 / a4;
    auto y6 = a1 / T{5};
    auto y7 = T{5} / a1;

    vector<Graph<T>> targets({a1, a2, a3, a4, a5});
    doValidate(y1, targets, res1, "division() Matrix-Matrix");
    doValidate(y2, targets, res2, "division() Matrix-Scalar");
    doValidate(y3, targets, res3, "division() Scalar-Matrix");
    doValidate(y4, targets, res4, "division() Dual Broadcasting");
    doValidate(y5, targets, res5, "division() Dual Broadcasting Reverse");
    doValidate(y6, targets, res6, "division() Matrix-Constant");
    doValidate(y7, targets, res7, "division() Constant-Matrix");
}

void testMax()
{
    Tensor<T> x  (Storage<T>({5, 7, 2, 1, -1, 3}), {2, 3});
    Tensor<T> y  (Storage<T>({2, 9, 2, 3, -2, 0}), {2, 3});
    Tensor<T> z  (Storage<T>({0}), {1});
    Tensor<T> w  (Storage<T>({3, 1}), {2, 1});
    Tensor<T> t  (Storage<T>({-1, 2, 3, 8}), {1, 4});

    Tensor<T> res1(Storage<T>({5, 9, 2, 3, -1, 3}), {2, 3});
    Tensor<T> res2(Storage<T>({5, 7, 2, 1, 0, 3}), {2, 3});
    Tensor<T> res3(Storage<T>({3, 3, 3, 8, 1, 2, 3, 8}), {2, 4});

    auto a  = make_variable<T>("a", x);
    auto b  = make_variable<T>("b", y);
    auto c  = make_variable<T>("c", z);
    auto d  = make_variable<T>("d", w);
    auto e  = make_variable<T>("e", t);

    auto y1 = max(a, b);
    auto y2 = max(a, c);
    auto y3 = max(c, a);
    auto y4 = max(d, e);
    auto y5 = max(e, d);

    vector<Graph<T>> targets({a, b, c, d, e});
    doValidate(y1, targets, res1, "max() Matrix-Matrix");
    doValidate(y2, targets, res2, "max() Matrix-Scalar");
    doValidate(y3, targets, res2, "max() Scalar-Matrix");
    doValidate(y4, targets, res3, "max() Dual Broadcasting");
    doValidate(y5, targets, res3, "max() Dual Broadcasting Reverse");
}

void testMin()
{
    Tensor<T> x  (Storage<T>({5, 7, 2, 1, -1, 3}), {2, 3});
    Tensor<T> y  (Storage<T>({2, 9, 2, 3, -2, 0}), {2, 3});
    Tensor<T> z  (Storage<T>({0}), {1});
    Tensor<T> w  (Storage<T>({3, 1}), {2, 1});
    Tensor<T> t  (Storage<T>({-1, 2, 3, 8}), {1, 4});

    Tensor<T> res1(Storage<T>({2, 7, 2, 1, -2, 0}), {2, 3});
    Tensor<T> res2(Storage<T>({0, 0, 0, 0, -1, 0}), {2, 3});
    Tensor<T> res3(Storage<T>({-1, 2, 3, 3, -1, 1, 1, 1}), {2, 4});

    auto a  = make_variable<T>("a", x);
    auto b  = make_variable<T>("b", y);
    auto c  = make_variable<T>("c", z);
    auto d  = make_variable<T>("d", w);
    auto e  = make_variable<T>("e", t);

    auto y1 = min(a, b);
    auto y2 = min(a, c);
    auto y3 = min(c, a);
    auto y4 = min(d, e);
    auto y5 = min(e, d);

    vector<Graph<T>> targets({a, b, c, d, e});
    doValidate(y1, targets, res1, "min() Matrix-Matrix");
    doValidate(y2, targets, res2, "min() Matrix-Scalar");
    doValidate(y3, targets, res2, "min() Scalar-Matrix");
    doValidate(y4, targets, res3, "min() Dual Broadcasting");
    doValidate(y5, targets, res3, "min() Dual Broadcasting Reverse");
}

void testReduceSum()
{
    Tensor<T> random(Storage<T>({0.01, 0.32, 0.74, 0.26, 0.38, 0.20}), {2, 3});

    auto a = make_variable<T>("a", range<T>({2, 3}));
    auto b = make_variable<T>("b", ones<T>({2, 3}));
    auto c = make_variable<T>("c", random);

    auto zero = make_constant<T>("0", Tensor<T>::fromScalar(0));
    auto one  = make_constant<T>("1", Tensor<T>::fromScalar(1));

    auto y1  = reduceSum(a);
    auto y2  = reduceSum(a, zero);
    auto y3  = reduceSum(a, one);
    auto y4  = reduceSum(b);
    auto y5  = reduceSum(b, zero);
    auto y6  = reduceSum(b, one);
    auto y7  = reduceSum(c);
    auto y8  = reduceSum(c, zero);
    auto y9  = reduceSum(c, one);

    vector<Graph<T>> targets({a, b, c});
    doValidate(y1, targets, scalar(15),               "reduceSum() Range");
    doValidate(y2, targets, rowVec(3, 5, 7),          "reduceSum() Range axis 0");
    doValidate(y3, targets, colVec(3, 12),            "reduceSum() Range axis 1");
    doValidate(y4, targets, scalar(6),                "reduceSum() Uniform");
    doValidate(y5, targets, rowVec(2, 2, 2),          "reduceSum() Uniform axis 0");
    doValidate(y6, targets, colVec(3, 3),             "reduceSum() Uniform axis 1");
    doValidate(y7, targets, scalar(1.91),             "reduceSum() Random");
    doValidate(y8, targets, rowVec(0.27, 0.70, 0.94), "reduceSum() Random axis 0");
    doValidate(y9, targets, colVec(1.07, 0.84),       "reduceSum() Random axis 1");
}

void testReduceProduct()
{
    Tensor<T> random(Storage<T>({0.01, 0.32, 0.74, 0.26, 0.38, 0.20}), {2, 3});

    auto a = make_variable<T>("a", range<T>({2, 3}));
    auto b = make_variable<T>("b", ones<T>({2, 3}));
    auto c = make_variable<T>("c", random);

    auto zero = make_constant<T>("0", Tensor<T>::fromScalar(0));
    auto one  = make_constant<T>("1", Tensor<T>::fromScalar(1));

    auto y1 = reduceProduct(a);
    auto y2 = reduceProduct(a, zero);
    auto y3 = reduceProduct(a, one);
    auto y4 = reduceProduct(b);
    auto y5 = reduceProduct(b, zero);
    auto y6 = reduceProduct(b, one);
    auto y7 = reduceProduct(c);
    auto y8 = reduceProduct(c, zero);
    auto y9 = reduceProduct(c, one);

    vector<Graph<T>> targets({a, b, c});
    doValidate(y1, targets, scalar(0),                "reduceProduct() Range");
    doValidate(y2, targets, rowVec(0, 4, 10),         "reduceProduct() Range axis 0");
    doValidate(y3, targets, colVec(0, 60),            "reduceProduct() Range axis 1");
    doValidate(y4, targets, scalar(1),                "reduceProduct() Uniform");
    doValidate(y5, targets, rowVec(1, 1, 1),          "reduceProduct() Uniform axis 0");
    doValidate(y6, targets, colVec(1, 1),             "reduceProduct() Uniform axis 1");
    doValidate(y7, targets, scalar(0.00),             "reduceProduct() Random");
    doValidate(y8, targets, rowVec(0.00, 0.12, 0.15), "reduceProduct() Random axis 0");
    doValidate(y9, targets, colVec(0.00, 0.02),       "reduceProduct() Random axis 1");
}

void testReduceMin()
{
    Tensor<T> random(Storage<T>({0.01, 0.32, 0.74, 0.26, 0.38, 0.20}), {2, 3});

    auto a = make_variable<T>("a", range<T>({2, 3}));
    auto b = make_variable<T>("b", ones<T>({2, 3}));
    auto c = make_variable<T>("c", random);

    auto zero = make_constant<T>("0", Tensor<T>::fromScalar(0));
    auto one  = make_constant<T>("1", Tensor<T>::fromScalar(1));

    auto y1 = reduceMin(a);
    auto y2 = reduceMin(a, zero);
    auto y3 = reduceMin(a, one);
    auto y4 = reduceMin(b);
    auto y5 = reduceMin(b, zero);
    auto y6 = reduceMin(b, one);
    auto y7 = reduceMin(c);
    auto y8 = reduceMin(c, zero);
    auto y9 = reduceMin(c, one);

    vector<Graph<T>> targets({a, b, c});
    doValidate(y1, targets, scalar(0),                "reduceMin() Range");
    doValidate(y2, targets, rowVec(0, 1, 2),          "reduceMin() Range axis 0");
    doValidate(y3, targets, colVec(0, 3),             "reduceMin() Range axis 1");
    // doValidate(y4, targets, scalar(1),                "reduceMin() Uniform"); <-- validate() does not correctly estimate the gradient
    // doValidate(y5, targets, rowVec(1, 1, 1),          "reduceMin() Uniform axis 0");
    // doValidate(y6, targets, colVec(1, 1),             "reduceMin() Uniform axis 1");
    doValidate(y7, targets, scalar(0.01),             "reduceMin() Random");
    doValidate(y8, targets, rowVec(0.01, 0.32, 0.20), "reduceMin() Random axis 0");
    doValidate(y9, targets, colVec(0.01, 0.20),       "reduceMin() Random axis 1");
}

void testReduceMax()
{
    Tensor<T> random(Storage<T>({0.01, 0.32, 0.74, 0.26, 0.38, 0.20}), {2, 3});

    auto a = make_variable<T>("a", range<T>({2, 3}));
    auto b = make_variable<T>("b", ones<T>({2, 3}));
    auto c = make_variable<T>("c", random);

    auto zero = make_constant<T>("0", Tensor<T>::fromScalar(0));
    auto one  = make_constant<T>("1", Tensor<T>::fromScalar(1));

    auto y1 = reduceMax(a);
    auto y2 = reduceMax(a, zero);
    auto y3 = reduceMax(a, one);
    auto y4 = reduceMax(b);
    auto y5 = reduceMax(b, zero);
    auto y6 = reduceMax(b, one);
    auto y7 = reduceMax(c);
    auto y8 = reduceMax(c, zero);
    auto y9 = reduceMax(c, one);

    vector<Graph<T>> targets({a, b, c});
    doValidate(y1, targets, scalar(5),                "reduceMax() Range");
    doValidate(y2, targets, rowVec(3, 4, 5),          "reduceMax() Range axis 0");
    doValidate(y3, targets, colVec(2, 5),             "reduceMax() Range axis 1");
    // doValidate(y4, targets, scalar(1),                "reduceMax() Uniform"); <-- validate() does not correctly estimate the gradient
    // doValidate(y5, targets, rowVec(1, 1, 1),          "reduceMax() Uniform axis 0");
    // doValidate(y6, targets, colVec(1, 1),             "reduceMax() Uniform axis 1");
    doValidate(y7, targets, scalar(0.74),             "reduceMax() Random");
    doValidate(y8, targets, rowVec(0.26, 0.38, 0.74), "reduceMax() Random axis 0");
    doValidate(y9, targets, colVec(0.74, 0.38),       "reduceMax() Random axis 1");
}

void testReduceMean()
{
    Tensor<T> random(Storage<T>({0.01, 0.32, 0.74, 0.26, 0.38, 0.20}), {2, 3});

    auto a = make_variable<T>("a", range<T>({2, 3}));
    auto b = make_variable<T>("b", ones<T>({2, 3}));
    auto c = make_variable<T>("c", random);

    auto zero = make_constant<T>("0", Tensor<T>::fromScalar(0));
    auto one  = make_constant<T>("1", Tensor<T>::fromScalar(1));

    auto y1 = reduceMean(a);
    auto y2 = reduceMean(a, zero);
    auto y3 = reduceMean(a, one);
    auto y4 = reduceMean(b);
    auto y5 = reduceMean(b, zero);
    auto y6 = reduceMean(b, one);
    auto y7 = reduceMean(c);
    auto y8 = reduceMean(c, zero);
    auto y9 = reduceMean(c, one);

    vector<Graph<T>> targets({a, b, c});
    doValidate(y1, targets, scalar(2.5),              "reduceMean() Range");
    doValidate(y2, targets, rowVec(1.5, 2.5, 3.5),    "reduceMean() Range axis 0");
    doValidate(y3, targets, colVec(1, 4),             "reduceMean() Range axis 1");
    doValidate(y4, targets, scalar(1),                "reduceMean() Uniform");
    doValidate(y5, targets, rowVec(1, 1, 1),          "reduceMean() Uniform axis 0");
    doValidate(y6, targets, colVec(1, 1),             "reduceMean() Uniform axis 1");
    doValidate(y7, targets, scalar(0.32),             "reduceMean() Random");
    doValidate(y8, targets, rowVec(0.14, 0.35, 0.47), "reduceMean() Random axis 0");
    doValidate(y9, targets, colVec(0.36, 0.28),       "reduceMean() Random axis 1");
}


int main()
{
    cout << "Op   Grad Name" << endl;

     testUnaryOps();       cout << endl;

     testAddition();       cout << endl;
     testSubtraction();    cout << endl;
     testMultiplication(); cout << endl;
     testDivision();       cout << endl;
     testMax();            cout << endl;
     testMin();            cout << endl;

     testReduceSum();      cout << endl;
     testReduceProduct();  cout << endl;
     testReduceMin();      cout << endl;
     testReduceMax();      cout << endl;
     testReduceMean();     cout << endl;


    return 0;
}

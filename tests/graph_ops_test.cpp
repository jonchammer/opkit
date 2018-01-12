#include <iostream>

#include "opkit/opkit.h"

using namespace std;
using namespace opkit;

using T = float;

template <class T>
bool operator==(const Tensor<T>& a, const Tensor<T>& b)
{
    if (a.shape() != b.shape()) return false;

    auto aIt = a.begin();
    auto bIt = b.begin();
    while (aIt != a.end())
    {
        if (abs(*aIt - *bIt) > 0.25) return false;
        ++aIt;
        ++bIt;
    }
    return true;
}

void doValidate(Graph<T> g, vector<Graph<T>>& targets,
    const Tensor<T>& expectedValue, const string& name)
{
    bool correctGradient = validate(g, targets, 1E-1);
    if (!(g() == expectedValue))
    {
        cout << "Expected Value: " << endl << expectedValue << endl;
        cout << "Actual Value:   " << endl << g()           << endl;
    }

    if (g() == expectedValue)
         cout << "[x]  ";
    else cout << "[ ]  ";

    if (correctGradient)
         cout << "[x]  ";
    else cout << "[ ]  ";

    cout << name << endl;
}

Tensor<T> scalar(T val)
{
    return Tensor<T>::fromScalar(val);
}

Tensor<T> matrix(initializer_list<T> values, const size_t M, const size_t N)
{
    return Tensor<T>::fromValues(values, {M, N});
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
    Tensor<T> x1 = matrix({5, 7, 2, 1, -1, 3}, 2, 3);
    Tensor<T> x2 = matrix({2, 9, 2, 3, -2, 0}, 2, 3);
    Tensor<T> x3 = scalar(1);
    Tensor<T> x4 = matrix({3, 1}, 2, 1);
    Tensor<T> x5 = matrix({-1, 2, 3, 8}, 1, 4);

    Tensor<T> res1 = matrix({7, 16, 4, 4, -3, 3}, 2, 3);
    Tensor<T> res2 = matrix({6, 8, 3, 2, 0, 4}, 2, 3);
    Tensor<T> res3 = matrix({2, 5, 6, 11, 0, 3, 4, 9}, 2, 4);
    Tensor<T> res4 = matrix({10, 12, 7, 6, 4, 8}, 2, 3);

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
    Tensor<T> random = Tensor<T>::fromValues({0.01, 0.32, 0.74, 0.26, 0.38, 0.20}, {2, 3});

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

    Tensor<T> res1 = scalar(15);
    Tensor<T> res2 = Tensor<T>::fromValues({3, 5, 7}, {1, 3});
    Tensor<T> res3 = Tensor<T>::fromValues({3, 12}, {2, 1});
    Tensor<T> res4 = scalar(6);
    Tensor<T> res5 = Tensor<T>::fromValues({2, 2, 2}, {1, 3});
    Tensor<T> res6 = Tensor<T>::fromValues({3, 3}, {2, 1});
    Tensor<T> res7 = scalar(1.91);
    Tensor<T> res8 = Tensor<T>::fromValues({0.27, 0.70, 0.94}, {1, 3});
    Tensor<T> res9 = Tensor<T>::fromValues({1.07, 0.84}, {2, 1});

    vector<Graph<T>> targets({a, b, c});
    doValidate(y1, targets, res1, "reduceSum() Range");
    doValidate(y2, targets, res2, "reduceSum() Range axis 0");
    doValidate(y3, targets, res3, "reduceSum() Range axis 1");
    doValidate(y4, targets, res4, "reduceSum() Uniform");
    doValidate(y5, targets, res5, "reduceSum() Uniform axis 0");
    doValidate(y6, targets, res6, "reduceSum() Uniform axis 1");
    doValidate(y7, targets, res7, "reduceSum() Random");
    doValidate(y8, targets, res8, "reduceSum() Random axis 0");
    doValidate(y9, targets, res9, "reduceSum() Random axis 1");
}

void testReduceProduct()
{
    Tensor<T> random = Tensor<T>::fromValues({0.01, 0.32, 0.74, 0.26, 0.38, 0.20}, {2, 3});

    auto a = make_variable<T>("a", range<T>({2, 3}));
    auto b = make_variable<T>("b", ones<T>({2, 3}));
    auto c = make_variable<T>("c", random);

    auto zero = make_constant<T>(0);
    auto one  = make_constant<T>(1);

    auto y1 = reduceProduct(a);
    auto y2 = reduceProduct(a, zero);
    auto y3 = reduceProduct(a, one);
    auto y4 = reduceProduct(b);
    auto y5 = reduceProduct(b, zero);
    auto y6 = reduceProduct(b, one);
    auto y7 = reduceProduct(c);
    auto y8 = reduceProduct(c, zero);
    auto y9 = reduceProduct(c, one);

    Tensor<T> res1 = scalar(0);
    Tensor<T> res2 = Tensor<T>::fromValues({0, 4, 10}, {1, 3});
    Tensor<T> res3 = Tensor<T>::fromValues({0, 60}, {2, 1});
    Tensor<T> res4 = scalar(1);
    Tensor<T> res5 = Tensor<T>::fromValues({1, 1, 1}, {1, 3});
    Tensor<T> res6 = Tensor<T>::fromValues({1, 1}, {2, 1});
    Tensor<T> res7 = scalar(0);
    Tensor<T> res8 = Tensor<T>::fromValues({0.00, 0.12, 0.15}, {1, 3});
    Tensor<T> res9 = Tensor<T>::fromValues({0.00, 0.02}, {2, 1});

    vector<Graph<T>> targets({a, b, c});
    doValidate(y1, targets, res1, "reduceProduct() Range");
    doValidate(y2, targets, res2, "reduceProduct() Range axis 0");
    doValidate(y3, targets, res3, "reduceProduct() Range axis 1");
    doValidate(y4, targets, res4, "reduceProduct() Uniform");
    doValidate(y5, targets, res5, "reduceProduct() Uniform axis 0");
    doValidate(y6, targets, res6, "reduceProduct() Uniform axis 1");
    doValidate(y7, targets, res7, "reduceProduct() Random");
    doValidate(y8, targets, res8, "reduceProduct() Random axis 0");
    doValidate(y9, targets, res9, "reduceProduct() Random axis 1");
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
    doValidate(y2, targets, Tensor<T>::fromValues({0, 1, 2}, {1, 3}),          "reduceMin() Range axis 0");
    doValidate(y3, targets, Tensor<T>::fromValues({0, 3}, {2, 1}),             "reduceMin() Range axis 1");
    // doValidate(y4, targets, scalar(1),                "reduceMin() Uniform"); <-- validate() does not correctly estimate the gradient
    // doValidate(y5, targets, rowVec(1, 1, 1),          "reduceMin() Uniform axis 0");
    // doValidate(y6, targets, colVec(1, 1),             "reduceMin() Uniform axis 1");
    doValidate(y7, targets, scalar(0.01),             "reduceMin() Random");
    doValidate(y8, targets, Tensor<T>::fromValues({0.01, 0.32, 0.20}, {1, 3}), "reduceMin() Random axis 0");
    doValidate(y9, targets, Tensor<T>::fromValues({0.01, 0.20}, {2, 1}),       "reduceMin() Random axis 1");
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
    doValidate(y2, targets, Tensor<T>::fromValues({3, 4, 5}, {1, 3}),          "reduceMax() Range axis 0");
    doValidate(y3, targets, Tensor<T>::fromValues({2, 5}, {2, 1}),             "reduceMax() Range axis 1");
    // doValidate(y4, targets, scalar(1),                "reduceMax() Uniform"); <-- validate() does not correctly estimate the gradient
    // doValidate(y5, targets, rowVec(1, 1, 1),          "reduceMax() Uniform axis 0");
    // doValidate(y6, targets, colVec(1, 1),             "reduceMax() Uniform axis 1");
    doValidate(y7, targets, scalar(0.74),             "reduceMax() Random");
    doValidate(y8, targets, Tensor<T>::fromValues({0.26, 0.38, 0.74}, {1, 3}), "reduceMax() Random axis 0");
    doValidate(y9, targets, Tensor<T>::fromValues({0.74, 0.38}, {2, 1}),       "reduceMax() Random axis 1");
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
    doValidate(y2, targets, Tensor<T>::fromValues({1.5, 2.5, 3.5}, {1, 3}),    "reduceMean() Range axis 0");
    doValidate(y3, targets, Tensor<T>::fromValues({1, 4}, {2, 1}),             "reduceMean() Range axis 1");
    doValidate(y4, targets, scalar(1),                "reduceMean() Uniform");
    doValidate(y5, targets, Tensor<T>::fromValues({1, 1, 1}, {1, 3}),          "reduceMean() Uniform axis 0");
    doValidate(y6, targets, Tensor<T>::fromValues({1, 1}, {2, 1}),             "reduceMean() Uniform axis 1");
    doValidate(y7, targets, scalar(0.32),             "reduceMean() Random");
    doValidate(y8, targets, Tensor<T>::fromValues({0.14, 0.35, 0.47}, {1, 3}), "reduceMean() Random axis 0");
    doValidate(y9, targets, Tensor<T>::fromValues({0.36, 0.28}, {2, 1}),       "reduceMean() Random axis 1");
}

void testNNOps()
{
    Tensor<T> random(Storage<T>({0.01, -0.32, 0.74, 0.26, -0.38, -0.20}), {2, 3});

    auto a = make_variable<T>("a", range<T>({2, 3}));
    auto b = make_variable<T>("b", ones<T>({2, 3}));
    auto c = make_variable<T>("c", random);

    auto y1  = relu(a);
    auto y2  = relu(b);
    auto y3  = relu(c);
    auto y4  = logistic(a);
    auto y5  = logistic(b);
    auto y6  = logistic(c);
    auto y7  = softplus(a);
    auto y8  = softplus(b);
    auto y9  = softplus(c);
    auto y10 = bentIdentity(a);
    auto y11 = bentIdentity(b);
    auto y12 = bentIdentity(c);
    auto y13 = softmax(a);
    auto y14 = softmax(b);
    auto y15 = softmax(c);

    Tensor<T> res1  = range<T>({2, 3});
    Tensor<T> res2  = ones<T>({2, 3});
    Tensor<T> res3  = Tensor<T>::fromValues({0.010,  0.000, 0.740, 0.260,  0.000,  0.000}, {2, 3});
    Tensor<T> res4  = Tensor<T>::fromValues({0.500,  0.731, 0.881, 0.953,  0.982,  0.993}, {2, 3});
    Tensor<T> res5  = Tensor<T>::fromValues({0.731,  0.731, 0.731, 0.731,  0.731,  0.731}, {2, 3});
    Tensor<T> res6  = Tensor<T>::fromValues({0.502,  0.421, 0.677, 0.565,  0.406,  0.450}, {2, 3});
    Tensor<T> res7  = Tensor<T>::fromValues({0.693,  1.313, 2.127, 3.049,  4.018,  5.007}, {2, 3});
    Tensor<T> res8  = Tensor<T>::fromValues({1.313,  1.313, 1.313, 1.313,  1.313,  1.313}, {2, 3});
    Tensor<T> res9  = Tensor<T>::fromValues({0.698,  0.546, 1.130, 0.832,  0.521,  0.598}, {2, 3});
    Tensor<T> res10 = Tensor<T>::fromValues({0.000,  1.207, 2.618, 4.081,  5.562,  7.050}, {2, 3});
    Tensor<T> res11 = Tensor<T>::fromValues({1.207,  1.207, 1.207, 1.207,  1.207,  1.207}, {2, 3});
    Tensor<T> res12 = Tensor<T>::fromValues({0.010, -0.295, 0.862, 0.277, -0.345, -0.190}, {2, 3});
    Tensor<T> res13 = Tensor<T>::fromValues({0.090,  0.245, 0.665, 0.090,  0.245,  0.665}, {2, 3});
    Tensor<T> res14 = Tensor<T>::fromValues({0.333,  0.333, 0.333, 0.333,  0.333,  0.333}, {2, 3});
    Tensor<T> res15 = Tensor<T>::fromValues({0.264,  0.189, 0.547, 0.463,  0.244,  0.292}, {2, 3});


    vector<Graph<T>> targets({a, b, c});
    doValidate(y1,  targets, res1,  "relu() Range");
    doValidate(y2,  targets, res2,  "relu() Ones");
    doValidate(y3,  targets, res3,  "relu() Random");
    doValidate(y4,  targets, res4,  "logistic() Range");
    doValidate(y5,  targets, res5,  "logistic() Ones");
    doValidate(y6,  targets, res6,  "logistic() Random");
    doValidate(y7,  targets, res7,  "softplus() Range");
    doValidate(y8,  targets, res8,  "softplus() Ones");
    doValidate(y9,  targets, res9,  "softplus() Random");
    doValidate(y10, targets, res10, "bentIdentity() Range");
    doValidate(y11, targets, res11, "bentIdentity() Ones");
    doValidate(y12, targets, res12, "bentIdentity() Random");
    doValidate(y13, targets, res13, "softmax() Range");
    doValidate(y14, targets, res14, "softmax() Ones");
    doValidate(y15, targets, res15, "softmax() Random");
}

void testMatrixMath()
{
    Storage<T> storage({-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});

    Tensor<T> x1 (storage, {3, 3});
    Tensor<T> x2 (storage, {3, 3}, {3, 2});
    Tensor<T> x3 (storage, {3, 3}, {6, 1});
    Tensor<T> x4 (storage, {3, 3}, {6, 2});
    Tensor<T> x5 (storage, {3, 2});
    Tensor<T> x6 (storage, {3, 2}, {2, 2});
    Tensor<T> x7 (storage, {3, 2}, {4, 1});
    Tensor<T> x8 (storage, {3, 2}, {4, 2});
    Tensor<T> x9 (storage, {3, 2}, {2, 0});
    Tensor<T> x10(storage, {3, 2}, {0, 1});
    Tensor<T> x11(storage, {3, 2}, {0, 0});
    Tensor<T> x12(storage, {2, 3});
    Tensor<T> x13(storage, {2, 3}, {3, 2});
    Tensor<T> x14(storage, {2, 3}, {6, 1});
    Tensor<T> x15(storage, {2, 3}, {6, 2});
    Tensor<T> x16(storage, {2, 3}, {3, 0});
    Tensor<T> x17(storage, {2, 3}, {0, 1});
    Tensor<T> x18(storage, {2, 3}, {0, 0});

    auto a1  = make_variable<T>("a1",   x1);
    auto a2  = make_variable<T>("a2",   x2);
    auto a3  = make_variable<T>("a3",   x3);
    auto a4  = make_variable<T>("a4",   x4);
    auto a5  = make_variable<T>("a5",   x5);
    auto a6  = make_variable<T>("a6",   x6);
    auto a7  = make_variable<T>("a7",   x7);
    auto a8  = make_variable<T>("a8",   x8);
    auto a9  = make_variable<T>("a9",   x9);
    auto a10 = make_variable<T>("a10", x10);
    auto a11 = make_variable<T>("a11", x11);
    auto a12 = make_variable<T>("a12", x12);
    auto a13 = make_variable<T>("a13", x13);
    auto a14 = make_variable<T>("a14", x14);
    auto a15 = make_variable<T>("a15", x15);
    auto a16 = make_variable<T>("a16", x16);
    auto a17 = make_variable<T>("a17", x17);
    auto a18 = make_variable<T>("a18", x18);
    vector<Graph<T>> targets({a1, a2, a3, a4, a5, a6, a7, a8, a9,
        a10, a11, a12, a13, a14, a15, a16, a17, a18});

    Tensor<T> res1  = Tensor<T>::fromValues({3, 0, -3, 12, 18, 24, 21, 36, 51}, {3, 3});
    Tensor<T> res2  = Tensor<T>::fromValues({3, -3, -9, 12, 24, 36, 21, 51, 81}, {3, 3});
    Tensor<T> res3  = Tensor<T>::fromValues({0, -3, -6, 36, 42, 48, 72, 87, 102}, {3, 3});
    Tensor<T> res4  = Tensor<T>::fromValues({0, -6, -12, 36, 48, 60, 72, 102, 132}, {3, 3});
    Tensor<T> res5  = Tensor<T>::fromValues({4, 1, 4, 10, 4, 19}, {3, 2});
    Tensor<T> res6  = Tensor<T>::fromValues({4, -2, 4, 16, 4, 34}, {3, 2});
    Tensor<T> res7  = Tensor<T>::fromValues({2, -1, 20, 26, 38, 53}, {3, 2});
    Tensor<T> res8  = Tensor<T>::fromValues({2, -4, 20, 32, 38, 68}, {3, 2});
    Tensor<T> res9  = Tensor<T>::fromValues({4, 4, 4, 4, 4, 4}, {3, 2});
    Tensor<T> res10 = Tensor<T>::fromValues({6, 3, -12, -6, -30, -15}, {3, 2});
    Tensor<T> res11 = Tensor<T>::fromValues({6, 6, -12, -12, -30, -30}, {3, 2});

    doValidate(matrixMultiply(a1, a1),  targets, res1,  "Matrix Multiply Square");
    doValidate(matrixMultiply(a1, a2),  targets, res2,  "Matrix Multiply Square X stride");
    doValidate(matrixMultiply(a1, a3),  targets, res3,  "Matrix multiply Square Y stride");
    doValidate(matrixMultiply(a1, a4),  targets, res4,  "Matrix Multiply Square X,Y stride");
    doValidate(matrixMultiply(a1, a5),  targets, res5,  "Matrix Multiply Rectangle");
    doValidate(matrixMultiply(a1, a6),  targets, res6,  "Matrix Multiply Rectangle X stride");
    doValidate(matrixMultiply(a1, a7),  targets, res7,  "Matrix multiply Rectangle Y stride");
    doValidate(matrixMultiply(a1, a8),  targets, res8,  "Matrix Multiply Rectangle X,Y stride");
    doValidate(matrixMultiply(a1, a9),  targets, res9,  "Matrix Multiply Rectangle 0 stride X");
    doValidate(matrixMultiply(a1, a10), targets, res10, "Matrix multiply Rectangle 0 stride Y");
    doValidate(matrixMultiply(a1, a11), targets, res11, "Matrix Multiply Rectangle 0 stride X,Y");

    Tensor<T> res12 = Tensor<T>::fromValues({21, 24, 27, 24, 30, 36, 27, 36, 45}, {3, 3});
    Tensor<T> res13 = Tensor<T>::fromValues({21, 27, 33, 24, 36, 48, 27, 45, 63}, {3, 3});
    Tensor<T> res14 = Tensor<T>::fromValues({48, 51, 54, 60, 66, 72, 72, 81, 90}, {3, 3});
    Tensor<T> res15 = Tensor<T>::fromValues({48, 54, 60, 60, 72, 84, 72, 90, 108}, {3, 3});
    Tensor<T> res16 = Tensor<T>::fromValues({12, 15, 12, 18, 12, 21}, {3, 2});
    Tensor<T> res17 = Tensor<T>::fromValues({12, 18, 12, 24, 12, 30}, {3, 2});
    Tensor<T> res18 = Tensor<T>::fromValues({30, 33, 36, 42, 42, 51}, {3, 2});
    Tensor<T> res19 = Tensor<T>::fromValues({30, 36, 36, 48, 42, 60}, {3, 2});
    Tensor<T> res20 = Tensor<T>::fromValues({12, 12, 12, 12, 12, 12}, {3, 2});
    Tensor<T> res21 = Tensor<T>::fromValues({-6, -3, -12, -6, -18, -9}, {3, 2});
    Tensor<T> res22 = Tensor<T>::fromValues({-6, -6, -12, -12, -18, -18}, {3, 2});

    doValidate(matrixMultiplyT1(a1, a1),  targets, res12, "Matrix Multiply T1 Square");
    doValidate(matrixMultiplyT1(a1, a2),  targets, res13, "Matrix Multiply T1 Square X stride");
    doValidate(matrixMultiplyT1(a1, a3),  targets, res14, "Matrix multiply T1 Square Y stride");
    doValidate(matrixMultiplyT1(a1, a4),  targets, res15, "Matrix Multiply T1 Square X,Y stride");
    doValidate(matrixMultiplyT1(a1, a5),  targets, res16, "Matrix Multiply T1 Rectangle");
    doValidate(matrixMultiplyT1(a1, a6),  targets, res17, "Matrix Multiply T1 Rectangle X stride");
    doValidate(matrixMultiplyT1(a1, a7),  targets, res18, "Matrix multiply T1 Rectangle Y stride");
    doValidate(matrixMultiplyT1(a1, a8),  targets, res19, "Matrix Multiply T1 Rectangle X,Y stride");
    doValidate(matrixMultiplyT1(a1, a9),  targets, res20, "Matrix Multiply T1 Rectangle 0 stride X");
    doValidate(matrixMultiplyT1(a1, a10), targets, res21, "Matrix multiply T1 Rectangle 0 stride Y");
    doValidate(matrixMultiplyT1(a1, a11), targets, res22, "Matrix Multiply T1 Rectangle 0 stride X,Y");

    Tensor<T> res23 = Tensor<T>::fromValues({5, -4, -13, -4, 14, 32, -13, 32, 77}, {3, 3});
    Tensor<T> res24 = Tensor<T>::fromValues({4, -5, -14, 4, 22, 40, 4, 49, 94}, {3, 3});
    Tensor<T> res25 = Tensor<T>::fromValues({5, -13, -31, -4, 32, 68, -13, 77, 167}, {3, 3});
    Tensor<T> res26 = Tensor<T>::fromValues({4, -14, -32, 4, 40, 76, 4, 94, 184}, {3, 3});
    Tensor<T> res27 = Tensor<T>::fromValues({5, -4, -4, 14, -13, 32}, {3, 2});
    Tensor<T> res28 = Tensor<T>::fromValues({4, -5, 4, 22, 4, 49}, {3, 2});
    Tensor<T> res29 = Tensor<T>::fromValues({5, -13, -4, 32, -13, 77}, {3, 2});
    Tensor<T> res30 = Tensor<T>::fromValues({4, -14, 4, 40, 4, 94}, {3, 2});
    Tensor<T> res31 = Tensor<T>::fromValues({6, -3, -12, 6, -30, 15}, {3, 2});
    Tensor<T> res32 = Tensor<T>::fromValues({5, 5, -4, -4, -13, -13}, {3, 2});
    Tensor<T> res33 = Tensor<T>::fromValues({6, 6, -12, -12, -30, -30}, {3, 2});

    doValidate(matrixMultiplyT2(a1, a1),  targets, res23, "Matrix Multiply T2 Square");
    doValidate(matrixMultiplyT2(a1, a2),  targets, res24, "Matrix Multiply T2 Square X stride");
    doValidate(matrixMultiplyT2(a1, a3),  targets, res25, "Matrix multiply T2 Square Y stride");
    doValidate(matrixMultiplyT2(a1, a4),  targets, res26, "Matrix Multiply T2 Square X,Y stride");
    doValidate(matrixMultiplyT2(a1, a12), targets, res27, "Matrix Multiply T2 Rectangle");
    doValidate(matrixMultiplyT2(a1, a13), targets, res28, "Matrix Multiply T2 Rectangle X stride");
    doValidate(matrixMultiplyT2(a1, a14), targets, res29, "Matrix multiply T2 Rectangle Y stride");
    doValidate(matrixMultiplyT2(a1, a15), targets, res30, "Matrix Multiply T2 Rectangle X,Y stride");
    doValidate(matrixMultiplyT2(a1, a16), targets, res31, "Matrix Multiply T2 Rectangle 0 stride X");
    doValidate(matrixMultiplyT2(a1, a17), targets, res32, "Matrix multiply T2 Rectangle 0 stride Y");
    doValidate(matrixMultiplyT2(a1, a18), targets, res33, "Matrix Multiply T2 Rectangle 0 stride X,Y");
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

    testNNOps();          cout << endl;
    testMatrixMath();     cout << endl;
    return 0;
}

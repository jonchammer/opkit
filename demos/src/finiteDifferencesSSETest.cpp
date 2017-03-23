/*
 * File:    finiteDifferencesSSETest.cpp
 * Author:  Jon C. Hammer
 * Purpose: This tests the "SSEFunction" class' default mechanism for calculating
 *          derivatives via finite differences. We calculate the value for a
 *          simple multivariate function whose Jacobian/Hessian matrices are
 *          known, and compare those to the finite differences approximations.
 *
 * Created on August 24, 2016, 9:03 PM
 */

#include <vector>
#include "opkit/opkit.h"

using namespace opkit;
using std::vector;

// This is a simple test function whose Jacobian and Hessian matrices (for each
// output) are known. This property allows us to compare the true Jacobian and
// Hessian against the finite differences approximation.
template <class T>
class TestFunction : public StandardFunction<T>
{
public:
    TestFunction() : StandardFunction<T>(2, 2, 6) {}
    using StandardFunction<T>::mParameters;

    void evaluate(const T* input, T* output)
    {
        output[0] = mParameters[0] * mParameters[0] * input[0] * input[0] + mParameters[1] * input[0] + mParameters[2];
        output[1] = mParameters[3] * mParameters[3] * input[1] * input[1] + mParameters[4] * input[1] + mParameters[5];
    }

    void calculateJacobianInputs(const T* x, Matrix<T>& jacobian)
    {
        jacobian.resize(2, 2);
        jacobian(0, 0) = T(2.0) * x[0] * mParameters[0] * mParameters[0] + mParameters[1];
        jacobian(0, 1) = T{};
        jacobian(1, 0) = T{};
        jacobian(1, 1) = T(2.0) * x[1] * mParameters[3] * mParameters[3] + mParameters[4];
    }

    void calculateJacobianParameters(const T* x, Matrix<T>& jacobian)
    {
        jacobian.resize(2, 6);
        jacobian.fill(T{});
        jacobian(0, 0) = T(2.0) * x[0] * x[0] * mParameters[0];
        jacobian(0, 1) = x[0];
        jacobian(0, 2) = T(1.0);
        jacobian(1, 3) = T(2.0) * x[1] * x[1] * mParameters[3];
        jacobian(1, 4) = x[1];
        jacobian(1, 5) = T(1.0);
    }

    void calculateHessianInputs(const T* /*x*/,
        const size_t outputIndex, Matrix<T>& hessian)
    {
        hessian.resize(2, 2);
        hessian.fill(T{});

        if (outputIndex == 0)
            hessian(0, 0) = T(2.0) * mParameters[0] * mParameters[0];
        else
            hessian(1, 1) = T(2.0) * mParameters[3] * mParameters[3];
    }

    void calculateHessianParameters(const T* x,
        const size_t outputIndex, Matrix<T>& hessian)
    {
        hessian.resize(6, 6);
        hessian.fill(T{});

        if (outputIndex == 0)
            hessian(0, 0) = T(2.0) * x[0] * x[0];
        else hessian(3, 3) = T(2.0) * x[1] * x[1];
    }
};

using Type = double;

int main()
{
    // Create a function
    TestFunction<Type> func;

    Rand rand(42);
    randomizeParameters(func.getParameters(), rand, 0.0, 0.1);

    // Create a synthetic dataset
    Matrix<Type> features(5, 2,
    {
         1.0,  2.0,
         0.5,  3.0,
         1.0,  3.0,
        -0.5,  1.0,
        -2.0, -1.0
    });

    Matrix<Type> labels(5, 3,
    {
         3.0,  2.0,  4.0,
         3.5,  1.5,  3.0,
         4.0,  3.0,  6.0,
         0.5, -0.5, -1.0,
        -3.0,  2.0,  4.0
    });

    SSEFunction<Type, TestFunction<Type>> errorFunction(func);

    // 1. Gradient with respect to parameters
    const size_t N = func.getNumParameters();
    vector<Type> gradientParameters1(N), gradientParameters2(N);
    errorFunction.calculateGradientParameters(features, labels, gradientParameters1);
    errorFunction.CostFunction::calculateGradientParameters(features, labels, gradientParameters2);

    for (size_t i = 0; i < gradientParameters1.size(); ++i)
    {
        if (abs(gradientParameters1[i] - gradientParameters2[i]) > 0.001)
        {
            cout << "Gradient Parameters - FAIL" << endl;
            printVector(cout, gradientParameters1);
            printVector(cout, gradientParameters2);
            return 1;
        }
    }
    cout << "Gradient Parameters - PASS" << endl;

    // 2. Gradient with respect to inputs
    const size_t M = func.getInputs();
    vector<Type> gradientInputs1(M), gradientInputs2(M);
    errorFunction.calculateGradientInputs(features, labels, gradientInputs1);
    errorFunction.CostFunction::calculateGradientInputs(features, labels, gradientInputs2);

    for (size_t i = 0; i < gradientInputs1.size(); ++i)
    {
        if (abs(gradientInputs1[i] - gradientInputs2[i]) > 0.001)
        {
            cout << "Gradient Inputs - FAIL" << endl;
            printVector(cout, gradientInputs1);
            printVector(cout, gradientInputs2);
            return 1;
        }
    }
    cout << "Gradient Inputs - PASS" << endl;

    // 3. Hessian with respect to parameters
    Matrix<Type> hessianParameters1(N, N);
    Matrix<Type> hessianParameters2(N, N);
    errorFunction.calculateHessianParameters(features, labels, hessianParameters1);
    errorFunction.CostFunction::calculateHessianParameters(features, labels, hessianParameters2);

    for (size_t i = 0; i < hessianParameters1.getRows(); ++i)
    for (size_t j = 0; j < hessianParameters1.getCols(); ++j)
    {
        if (abs(hessianParameters1(i, j) - hessianParameters2(i, j)) > 0.001)
        {
            cout << "Hessian Parameters - FAIL" << endl;
            printMatrix(cout, hessianParameters1); cout << endl;
            printMatrix(cout, hessianParameters2);
            return 1;
        }
    }
    cout << "Hessian Parameters - PASS" << endl;

    // 4. Hessian with respect to inputs
    Matrix<Type> hessianInputs1(M, M);
    Matrix<Type> hessianInputs2(M, M);
    errorFunction.calculateHessianInputs(features, labels, hessianInputs1);
    errorFunction.CostFunction::calculateHessianInputs(features, labels, hessianInputs2);

    for (size_t i = 0; i < hessianInputs1.getRows(); ++i)
    for (size_t j = 0; j < hessianInputs1.getCols(); ++j)
    {
        if (abs(hessianInputs1(i, j) - hessianInputs2(i, j)) > 0.001)
        {
            cout << "Hessian Inputs - FAIL" << endl;
            printMatrix(cout, hessianInputs1);
            printMatrix(cout, hessianInputs2);
            return 1;
        }
    }
    cout << "Hessian Inputs - PASS" << endl;
    return 0;
}

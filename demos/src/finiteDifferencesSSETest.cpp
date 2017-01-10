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
#include "Function.h"
#include "PrettyPrinter.h"
#include "Dataset.h"
#include "Matrix.h"
#include "SSEFunction.h"
#include "CommonFunctions.h"

using namespace opkit;
using std::vector;

int main()
{
    // Create a function
    MultivariateLinear<double> func(2, 3);
    randomizeParameters(func.getParameters(), 0.0, 0.1);

    // Create a synthetic dataset
    Dataset<double> features, labels;
    features.setSize(5, 2);
    labels.setSize(5, 3);
    features.row(0) = {1.0, 2.0};   labels.row(0) = {3.0, 2.0, 4.0};
    features.row(1) = {0.5, 3.0};   labels.row(1) = {3.5, 1.5, 3.0};
    features.row(2) = {1.0, 3.0};   labels.row(2) = {4.0, 3.0, 6.0};
    features.row(3) = {-0.5, 1.0};  labels.row(3) = {0.5, -0.5, -1.0};
    features.row(4) = {-2.0, -1.0}; labels.row(4) = {-3.0, 2.0, 4.0};

    SSEFunction<double, MultivariateLinear<double>> errorFunction(func);

    // 1. Gradient with respect to parameters
    const size_t N = func.getNumParameters();
    vector<double> gradientParameters1(N), gradientParameters2(N);
    errorFunction.calculateGradientParameters(features, labels, gradientParameters1);
    errorFunction.ErrorFunction::calculateGradientParameters(features, labels, gradientParameters2);

    for (size_t i = 0; i < gradientParameters1.size(); ++i)
    {
        if (abs(gradientParameters1[i] - gradientParameters2[i]) > 0.001)
        {
            cout << "Gradient Parameters - FAIL" << endl;
            printVector(gradientParameters1);
            printVector(gradientParameters2);
            return 1;
        }
    }
    cout << "Gradient Parameters - PASS" << endl;

    // 2. Gradient with respect to inputs
    const size_t M = func.getInputs();
    vector<double> gradientInputs1(M), gradientInputs2(M);
    errorFunction.calculateGradientInputs(features, labels, gradientInputs1);
    errorFunction.ErrorFunction::calculateGradientInputs(features, labels, gradientInputs2);

    for (size_t i = 0; i < gradientInputs1.size(); ++i)
    {
        if (abs(gradientInputs1[i] - gradientInputs2[i]) > 0.001)
        {
            cout << "Gradient Inputs - FAIL" << endl;
            printVector(gradientInputs1);
            printVector(gradientInputs2);
            return 1;
        }
    }
    cout << "Gradient Inputs - PASS" << endl;

    // 3. Hessian with respect to parameters
    Matrix<double> hessianParameters1(N, N);
    Matrix<double> hessianParameters2(N, N);
    errorFunction.calculateHessianParameters(features, labels, hessianParameters1);
    errorFunction.ErrorFunction::calculateHessianParameters(features, labels, hessianParameters2);

    for (size_t i = 0; i < hessianParameters1.getRows(); ++i)
    for (size_t j = 0; j < hessianParameters1.getCols(); ++j)
    {
        if (abs(hessianParameters1(i, j) - hessianParameters2(i, j)) > 0.001)
        {
            cout << "Hessian Parameters - FAIL" << endl;
            printMatrix(hessianParameters1);
            printMatrix(hessianParameters2);
            return 1;
        }
    }
    cout << "Hessian Parameters - PASS" << endl;

    // 4. Hessian with respect to inputs
    Matrix<double> hessianInputs1(M, M);
    Matrix<double> hessianInputs2(M, M);
    errorFunction.calculateHessianInputs(features, labels, hessianInputs1);
    errorFunction.ErrorFunction::calculateHessianInputs(features, labels, hessianInputs2);

    for (size_t i = 0; i < hessianInputs1.getRows(); ++i)
    for (size_t j = 0; j < hessianInputs1.getCols(); ++j)
    {
        if (abs(hessianInputs1(i, j) - hessianInputs2(i, j)) > 0.001)
        {
            cout << "Hessian Inputs - FAIL" << endl;
            printMatrix(hessianInputs1);
            printMatrix(hessianInputs2);
            return 1;
        }
    }
    cout << "Hessian Inputs - PASS" << endl;

    return 0;
}

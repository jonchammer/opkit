/*
 * File:    finiteDifferencesTest.cpp
 * Author:  Jon C. Hammer
 * Purpose: This tests the "Function" class' default mechanism for calculating
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
class TestFunction : public StandardFunction<double>
{
public:
    TestFunction() : StandardFunction(2, 2, 6) {}

    void evaluate(const vector<double>& input, vector<double>& output)
    {
        output.resize(2);
        output[0] = mParameters[0] * mParameters[0] * input[0] * input[0] + mParameters[1] * input[0] + mParameters[2];
        output[1] = mParameters[3] * mParameters[3] * input[1] * input[1] + mParameters[4] * input[1] + mParameters[5];
    }

    void calculateJacobianInputs(const vector<double>& x, Matrix<double>& jacobian)
    {
        jacobian.resize(2, 2);
        jacobian(0, 0) = 2 * x[0] * mParameters[0] * mParameters[0] + mParameters[1];
        jacobian(0, 1) = 0.0;
        jacobian(1, 0) = 0.0;
        jacobian(1, 1) = 2 * x[1] * mParameters[3] * mParameters[3] + mParameters[4];
    }

    void calculateJacobianParameters(const vector<double>& x, Matrix<double>& jacobian)
    {
        jacobian.resize(2, 6);
        jacobian.fill(0.0);
        jacobian(0, 0) = 2.0 * x[0] * x[0] * mParameters[0];
        jacobian(0, 1) = x[0];
        jacobian(0, 2) = 1;
        jacobian(1, 3) = 2.0 * x[1] * x[1] * mParameters[3];
        jacobian(1, 4) = x[1];
        jacobian(1, 5) = 1;
    }

    void calculateHessianInputs(const vector<double>& /*x*/,
        const size_t outputIndex, Matrix<double>& hessian)
    {
        hessian.resize(2, 2);
        hessian.fill(0.0);

        if (outputIndex == 0)
            hessian(0, 0) = 2 * mParameters[0] * mParameters[0];
        else
            hessian(1, 1) = 2 * mParameters[3] * mParameters[3];
    }

    void calculateHessianParameters(const vector<double>& x,
        const size_t outputIndex, Matrix<double>& hessian)
    {
        hessian.resize(6, 6);
        hessian.fill(0.0);

        if (outputIndex == 0)
            hessian(0, 0) = 2 * x[0] * x[0];
        else hessian(3, 3) = 2 * x[1] * x[1];
    }
};

int main()
{
    TestFunction f;
    vector<double> input = {2.0, 3.0};

    vector<double>& params = f.getParameters();
    params[0] = 1.0;
    params[1] = 2.0;
    params[2] = 3.0;
    params[3] = 4.0;
    params[4] = 5.0;
    params[5] = 6.0;

    // 1. Jacobian with respect to parameters
    Matrix<double> jacobianParameters1, jacobianParameters2;
    f.calculateJacobianParameters(input, jacobianParameters1);
    f.Function::calculateJacobianParameters(input, jacobianParameters2);

    for (size_t i = 0; i < jacobianParameters1.getRows(); ++i)
    {
        for (size_t j = 0; j < jacobianParameters1.getCols(); ++j)
        {
            if (abs(jacobianParameters1(i, j) - jacobianParameters2(i, j)) > 0.001)
            {
                cout << "Jacobian Parameters - FAIL" << endl;
                cout << "Truth:" << endl;
                printMatrix(jacobianParameters1);

                cout << "Finite Differences:" << endl;
                printMatrix(jacobianParameters2);
                return 1;
            }
        }
    }
    cout << "Jacobian Parameters - PASS" << endl;

    // 2. Jacobian with respect to inputs
    Matrix<double> jacobianInputs1, jacobianInputs2;
    f.calculateJacobianInputs(input, jacobianInputs1);
    f.Function::calculateJacobianInputs(input, jacobianInputs2);

    for (size_t i = 0; i < jacobianInputs1.getRows(); ++i)
    {
        for (size_t j = 0; j < jacobianInputs1.getCols(); ++j)
        {
            if (abs(jacobianInputs1(i, j) - jacobianInputs2(i, j)) > 0.001)
            {
                cout << "Jacobian Inputs - FAIL" << endl;
                cout << "Truth:" << endl;
                printMatrix(jacobianInputs1);

                cout << "Finite Differences:" << endl;
                printMatrix(jacobianInputs2);
                return 1;
            }
        }
    }
    cout << "Jacobian Inputs - PASS" << endl;

    // 3. Hessian(index) with respect to parameters
    for (size_t index = 0; index < f.getOutputs(); ++index)
    {
        Matrix<double> hessianParameters1, hessianParameters2;
        f.calculateHessianParameters(input, index, hessianParameters1);
        f.Function::calculateHessianParameters(input, index, hessianParameters2);

        cout << "Hessian Parameters - Output: " << index << endl;
        for (size_t i = 0; i < hessianParameters1.getRows(); ++i)
        {
            for (size_t j = 0; j < hessianParameters1.getCols(); ++j)
            {
                if (abs(hessianParameters1(i, j) - hessianParameters2(i, j)) > 0.001)
                {
                    cout << "Hessian Parameters - FAIL" << endl;
                    cout << "Truth:" << endl;
                    printMatrix(hessianParameters1);

                    cout << "Finite Differences:" << endl;
                    printMatrix(hessianParameters2);
                    return 1;
                }
            }
        }
        cout << "Hessian Parameters - PASS" << endl;
    }

        // 4. Hessian(index) with respect to inputs
    for (size_t index = 0; index < f.getOutputs(); ++index)
    {
        Matrix<double> hessianInputs1, hessianInputs2;
        f.calculateHessianInputs(input, index, hessianInputs1);
        f.Function::calculateHessianInputs(input, index, hessianInputs2);

        cout << "Hessian - Input: " << index << endl;
        for (size_t i = 0; i < hessianInputs1.getRows(); ++i)
        {
            for (size_t j = 0; j < hessianInputs1.getCols(); ++j)
            {
                if (abs(hessianInputs1(i, j) - hessianInputs2(i, j)) > 0.001)
                {
                    cout << "Hessian Inputs - FAIL" << endl;
                    cout << "Truth:" << endl;
                    printMatrix(hessianInputs1);

                    cout << "Finite Differences:" << endl;
                    printMatrix(hessianInputs2);
                    return 1;
                }
            }
        }
        cout << "Hessian Inputs - PASS" << endl;
    }

    return 0;
}

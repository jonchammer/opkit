/*
 * File:   Functions.h
 * Author: Jon C. Hammer
 *
 * Created on July 10, 2016, 12:03 PM
 */

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <vector>
#include "Function.h"
#include "Dataset.h"
using std::vector;

namespace opkit
{

// This is a model representing a standard 2D line: y = mx + b
template <class T>
class SimpleLinear : public StandardFunction<T>
{
public:
    SimpleLinear() : StandardFunction<T>(1, 1, 2) {}

    void evaluate(const vector<T>& input, vector<T>& output)
    {
        output.resize(StandardFunction<T>::mOutputs);

        T m = StandardFunction<T>::mParameters[0];
        T b = StandardFunction<T>::mParameters[1];

        output[0] = m * input[0] + b;
    }

    void calculateJacobianInputs(const vector<T>& /*x*/, Dataset<T>& jacobian)
    {
        jacobian.setSize(1, 1);
        jacobian[0][0] = StandardFunction<T>::mParameters[0];
    }

    void calculateJacobianParameters(const vector<T>& x, Dataset<T>& jacobian)
    {
        jacobian.setSize(1, 2);
        jacobian[0][0] = x[0];
        jacobian[0][1] = 1;
    }
};

// This is a model representing a standard 2D quadratic: y = ax^2 + bx + c
template <class T>
class SimpleQuadratic : public StandardFunction<T>
{
public:
    SimpleQuadratic() : StandardFunction<T>(1, 1, 3) {}

    void evaluate(const vector<T>& input, vector<T>& output)
    {
        output.resize(StandardFunction<T>::mOutputs);

        T a = StandardFunction<T>::mParameters[0];
        T b = StandardFunction<T>::mParameters[1];
        T c = StandardFunction<T>::mParameters[2];

        output[0] = (a * input[0] * input[0]) + (b * input[0]) + c;
    }

    void calculateJacobianInputs(const vector<T>& x, Dataset<T>& jacobian)
    {
        jacobian.setSize(1, 1);
        jacobian[0][0] = 2 * StandardFunction<T>::mParameters[0] * x[0] + StandardFunction<T>::mParameters[1];
    }

    void calculateJacobianParameters(const vector<T>& x, Dataset<T>& jacobian)
    {
        jacobian.setSize(1, 3);
        jacobian[0][0] = x[0] * x[0];
        jacobian[0][1] = x[0];
        jacobian[0][2] = 1;
    }
};

// This is a model representing an arbitrary 2D polynomial of the form
// y = ax^N + bx^N-1 + c^N-2 + ... pN + q. The degree of the polynomial
// is provided by the user.
template <class T>
class SimplePolynomial : public StandardFunction<T>
{
public:
    SimplePolynomial(int degree) : StandardFunction<T>(1, 1, degree + 1) {}

    void evaluate(const vector<T>& input, vector<T>& output)
    {
        output.resize(1);

        T sum = 1.0;
        output[0]  = 0.0;
        for (int i = StandardFunction<T>::mParameters.size() - 1; i >= 0; --i)
        {
            output[0] += StandardFunction<T>::mParameters[i] * sum;
            sum       *= input[0];
        }
    }
};

// This is a model representing a multivariate linear model of the form
// y = Mx + b, where M is a Dataset of weights (input rows x output cols),
// x is a vector (number of inputs), b is a vector (number of outputs),
// and y is a vector (number of outputs).
//
// The parameters for the Dataset are stored first, followed by the parameters
// that represent the biases. For example, if we have 3 inputs and 2 outputs,
// the Dataset looks like this:
//   [w11 w12]
//   [w21 w22]
//   [w31 w32]
// and the parameters will be organized like this:
//   [w11 w12] [w21 w22] [w31 w32] [b1] [b2]
//
// The Dataset weights represent the strength of the connection between the
// ith input and the jth output (wij).
template <class T>
class MultivariateLinear : public StandardFunction<T>
{
public:
    MultivariateLinear(int inputs, int outputs) :
        StandardFunction<T>(inputs, outputs, inputs * outputs + outputs) {}

    void evaluate(const vector<T>& input, vector<T>& output)
    {
        output.resize(StandardFunction<T>::mOutputs);
        std::fill(output.begin(), output.end(), 0.0);

        // The biases are the last 'mOutputs' terms in the parameter list
        int biasStart = StandardFunction<T>::mParameters.size() - StandardFunction<T>::mOutputs;

        for (size_t j = 0; j < StandardFunction<T>::mOutputs; ++j)
        {
            // M*x
            for (size_t i = 0; i < StandardFunction<T>::mInputs; ++i)
                output[j] += StandardFunction<T>::mParameters[i * StandardFunction<T>::mOutputs + j] * input[i];

            // + b
            output[j] += StandardFunction<T>::mParameters[biasStart + j];
        }
    }
};

};
#endif /* FUNCTIONS_H */

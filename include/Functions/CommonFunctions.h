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
using std::vector;

namespace athena
{
    
// This is a model representing a standard 2D line: y = mx + b
class SimpleLinear : public StandardFunction
{
public:
    SimpleLinear() : StandardFunction(1, 1, 2) {}

    void evaluate(const vector<double>& input, vector<double>& output)
    {
        output.resize(mOutputs);
        
        double m = mParameters[0];
        double b = mParameters[1];
        
        output[0] = m * input[0] + b;
    }
    
    void calculateJacobianInputs(const vector<double>& /*x*/, Matrix& jacobian)
    {
        jacobian.setSize(1, 1);
        jacobian[0][0] = mParameters[0];
    }
    
    void calculateJacobianParameters(const vector<double>& x, Matrix& jacobian)
    {
        jacobian.setSize(1, 2);
        jacobian[0][0] = x[0];
        jacobian[0][1] = 1;
    }
};

// This is a model representing a standard 2D quadratic: y = ax^2 + bx + c
class SimpleQuadratic : public StandardFunction
{
public:
    SimpleQuadratic() : StandardFunction(1, 1, 3) {}
    
    void evaluate(const vector<double>& input, vector<double>& output)
    {
        output.resize(mOutputs);
        
        double a = mParameters[0];
        double b = mParameters[1];
        double c = mParameters[2];
        
        output[0] = (a * input[0] * input[0]) + (b * input[0]) + c;
    }
    
    void calculateJacobianInputs(const vector<double>& x, Matrix& jacobian)
    {
        jacobian.setSize(1, 1);
        jacobian[0][0] = 2 * mParameters[0] * x[0] + mParameters[1];
    }
    
    void calculateJacobianParameters(const vector<double>& x, Matrix& jacobian)
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
class SimplePolynomial : public StandardFunction
{
public:
    SimplePolynomial(int degree) : StandardFunction(1, 1, degree + 1) {}
    
    void evaluate(const vector<double>& input, vector<double>& output)
    {
        output.resize(1);

        double sum = 1;
        output[0]  = 0.0;
        for (int i = mParameters.size() - 1; i >= 0; --i)
        {
            output[0] += mParameters[i] * sum;
            sum       *= input[0];
        }
    }
};

// This is a model representing a multivariate linear model of the form
// y = Mx + b, where M is a matrix of weights (input rows x output cols),
// x is a vector (number of inputs), b is a vector (number of outputs),
// and y is a vector (number of outputs).
//
// The parameters for the matrix are stored first, followed by the parameters
// that represent the biases. For example, if we have 3 inputs and 2 outputs,
// the matrix looks like this:
//   [w11 w12]
//   [w21 w22]
//   [w31 w32]
// and the parameters will be organized like this:
//   [w11 w12] [w21 w22] [w31 w32] [b1] [b2]
//
// The matrix weights represent the strength of the connection between the
// ith input and the jth output (wij).
class MultivariateLinear : public StandardFunction
{
public:
    MultivariateLinear(int inputs, int outputs) : 
        StandardFunction(inputs, outputs, inputs * outputs + outputs) {}
    
    void evaluate(const vector<double>& input, vector<double>& output)
    {
        output.resize(mOutputs);
        std::fill(output.begin(), output.end(), 0.0);
        
        // The biases are the last 'mOutputs' terms in the parameter list
        int biasStart = mParameters.size() - mOutputs;
        
        for (size_t j = 0; j < mOutputs; ++j)
        {
            // M*x
            for (size_t i = 0; i < mInputs; ++i)
                output[j] += mParameters[i * mOutputs + j] * input[i];
            
            // + b
            output[j] += mParameters[biasStart + j];
        }
    }
};

};
#endif /* FUNCTIONS_H */


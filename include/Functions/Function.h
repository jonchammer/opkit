/*
 * File:   Model.h
 * Author: Jon C. Hammer
 *
 * Created on July 9, 2016, 7:56 PM
 */

#ifndef MODEL_H
#define MODEL_H

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include "Dataset.h"
#include "Error.h"

using std::cout;
using std::endl;
using std::vector;

namespace opkit
{
template <class T>
class Function
{
public:
    // ------------------------- Interface Methods ------------------------- //

    // Apply this function to the given input in order to produce an output.
    // That output will be stored in 'output'.
    virtual void evaluate(const vector<T>& input, vector<T>& output) = 0;

    // Returns the number of inputs to the function and the number of outputs,
    // respectively.
    virtual size_t getInputs()  const = 0;
    virtual size_t getOutputs() const = 0;

    // Functions may be parameterized (i.e. have variables in the place of some
    // constants). Parameterized functions can be thought of as generalized
    // versions of traditional concrete functions. These functions return the
    // current parameters and the number of parameters, respectively.
    //
    // An example of a parameterized function is the linear model: y = mx + b.
    // Here, m and b are the parameters to the function.
    virtual vector<T>& getParameters()             = 0;
    virtual const vector<T>& getParameters() const = 0;
    virtual size_t getNumParameters() const        = 0;

    // ---------------------- Default Implementations ---------------------- //

    virtual ~Function() {}

    // This function should return true when the implementing function is
    // capable of saving the result of the last evaluation. By default, this
    // method returns false.
    virtual bool cachesLastEvaluation() const;

    // For functions in which 'cachesLastEvaluation' returns true, this method
    // is used to return the cached value. This is helpful for complicated
    // functions (for which a single evaluation can be expensive).
    virtual void getLastEvaluation(vector<T>& output);

    // Calculates the Jacobian of this function df(x)/dx with respect to the
    // function inputs. 'x' is the point at which the Jacobian should be
    // calculated, and the Jacobian itself is stored in 'Jacobian'.
    virtual void calculateJacobianInputs(const vector<T>& x, Dataset<T>& jacobian);

    // Calculates the Jacobian of this function df(x)/dx with respect to the
    // function parameters. 'x' is the point at which the Jacobian should be
    // calculated, and the Jacobian itself is stored in 'Jacobian'.
    virtual void calculateJacobianParameters(const vector<T>& x, Dataset<T>& jacobian);

    // Calculates the Hessian Dataset of this function with respect to the
    // function inputs. 'x' is the point at which the Hessian should be
    // calculated, and the Hessian itself is stored in 'hessian'. Since an
    // arbitrary function can have many inputs and outputs, the second
    // derivative is technically a 3rd order tensor (3D Dataset). This function
    // will only calculate the Hessian with respect to a single output, indexed
    // by 'outputIndex'. For scalar functions, 'outputIndex' should be 0.
    virtual void calculateHessianInputs(const vector<T>& x,
        const size_t outputIndex, Dataset<T>& hessian);

    // Calculates the Hessian Dataset of this function with respect to the
    // function parameters. 'x' is the point at which the Hessian should be
    // calculated, and the Hessian itself is stored in 'hessian'. Since an
    // arbitrary function can have many inputs and outputs, the second
    // derivative is technically a 3rd order tensor (3D Dataset). This function
    // will only calculate the Hessian with respect to a single output, indexed
    // by 'outputIndex'. For scalar functions, 'outputIndex' should be 0.
    virtual void calculateHessianParameters(const vector<T>& x,
        const size_t outputIndex, Dataset<T>& hessian);
};

// Most functions will maintain a vector of parameters. They can inherit from
// this class in order to provide a default implementation for most functions.
template <class T>
class StandardFunction : public Function<T>
{
public:
    StandardFunction(size_t inputs, size_t outputs, size_t numParams) :
        mInputs(inputs), mOutputs(outputs), mParameters(numParams) {}

    // Getters
    size_t getInputs() const               { return mInputs;            }
    size_t getOutputs() const              { return mOutputs;           }
    vector<T>& getParameters()             { return mParameters;        }
    const vector<T>& getParameters() const { return mParameters;        }
    size_t getNumParameters() const        { return mParameters.size(); }

protected:
    size_t mInputs, mOutputs;
    vector<T> mParameters;
};

template <class T>
bool Function<T>::cachesLastEvaluation() const
{
    return false;
}

template <class T>
void Function<T>::getLastEvaluation(vector<T>& output)
{
    throw Ex("Function::getLastEvaluation not implemented.");
}

template <class T>
void Function<T>::calculateJacobianInputs(const vector<T>& x, Dataset<T>& jacobian)
{
    cout << "Function::calculateJacobianInputs()" << endl;

    // Constants used in the finite differences approximation
    // Epsilon needs to be a reasonably small number, but the size depends on
    // the type (e.g. doubles need smaller values). We use the sqrt of the
    // machine epsilon as a good starting point.
    const static T EPSILON = std::sqrt(std::numeric_limits<T>::epsilon());
    const size_t N         = getInputs();
    const size_t M         = getOutputs();

    // Ensure the Jacobian Dataset is large enough
    jacobian.setSize(M, N);

    // Temporary vectors used for calculations
    static vector<T> prediction(M, T{});
    static vector<T> derivativePrediction(M, T{});
    static vector<T> input(N, T{});

    // Start by evaluating the function without any modifications
    std::copy(x.begin(), x.end(), input.begin());
    evaluate(input, prediction);

    // The Jacobian is calculated one column at a time by changing one input
    // and measuring the effect on all M outputs.
    for (size_t p = 0; p < N; ++p)
    {
        // Save the original value of this input
        T orig = input[p];

        // Calculate the derivative of the function (y) with respect to
        // the current input, p, by slightly changing that input
        // and measuring comparing the output that with no change applied.
        input[p] += EPSILON;
        evaluate(input, derivativePrediction);

        for (size_t r = 0; r < M; ++r)
            jacobian[r][p] = (derivativePrediction[r] - prediction[r]) / EPSILON;

        // Change the input back to its original value
        input[p] = orig;
    }
}

template <class T>
void Function<T>::calculateJacobianParameters(const vector<T>& x, Dataset<T>& jacobian)
{
    cout << "Function::calculateJacobianParameters()" << endl;

    // Constants used in the finite differences approximation
    // Epsilon needs to be a reasonably small number, but the size depends on
    // the type (e.g. doubles need smaller values). We use the sqrt of the
    // machine epsilon as a good starting point.
    const static T EPSILON = std::sqrt(std::numeric_limits<T>::epsilon());
    const size_t N         = getNumParameters();
    const size_t M         = getOutputs();

    // Ensure the Jacobian Dataset is large enough
    jacobian.setSize(M, N);

    // Temporary vectors used for calculations
    static vector<T> prediction(M, 0.0);
    static vector<T> derivativePrediction(M, 0.0);

    // Start by evaluating the function without any modifications
    vector<T>& parameters = getParameters();
    evaluate(x, prediction);

    for (size_t p = 0; p < N; ++p)
    {
        // Save the original value of this parameter
        T orig = parameters[p];

        // Calculate the derivative of the function (y) with respect to
        // the current parameter, p, by slightly changing that parameter
        // and measuring comparing the output that with no change applied.
        parameters[p] += EPSILON;
        evaluate(x, derivativePrediction);

        for (size_t r = 0; r < M; ++r)
            jacobian[r][p] = (derivativePrediction[r] - prediction[r]) / EPSILON;

        // Change the parameter back to its original value
        parameters[p] = orig;
    }
}

template <class T>
void Function<T>::calculateHessianInputs(const vector<T>& x,
        const size_t outputIndex, Dataset<T>& hessian)
{
    cout << "Function::calculateHessianInputs()" << endl;

    // Epsilon has to be set to a larger value than that used in calculating
    // the gradient because it will be squared in the calculations below. If it
    // is too small, we incur more significant rounding errors.
    const static T EPSILON = std::pow(std::numeric_limits<T>::epsilon(), T{0.25});
    const size_t N         = getInputs();
    const size_t M         = getOutputs();
    hessian.setSize(N, N);

    // Create the temporary vectors we'll need
    static vector<T> base(M, T{});
    static vector<T> ei(M, T{});
    static vector<T> ej(M, T{});
    static vector<T> eij(M, T{});
    static vector<T> input(N, T{});

    // Perform one evaluation with no changes to get a baseline measurement
    std::copy(x.begin(), x.end(), input.begin());
    evaluate(x, base);

    // Using the method of finite differences, each element of the Hessian
    // can be approximated using the following formula:
    // H(i,j) = (f(x1,x2,...xi + h, ...xj + k...xn) - f(x1, x2 ,...xi + h...xn)
    //          - f(x1, x2, ... xj + k ... xn) + f(x1...xn)) / hk
    for (size_t i = 0; i < N; ++i)
    {
        // Modify i alone
        T origI      = input[i];
        input[i]    += EPSILON;
        evaluate(input, ei);
        input[i]     = origI;

        for (size_t j = 0; j < N; ++j)
        {
            // Modify i and j
            T origJ      = input[j];
            input[i]    += EPSILON;
            input[j]    += EPSILON;
            evaluate(input, eij);
            input[i]     = origI;
            input[j]     = origJ;

            // Modify j alone
            input[j] += EPSILON;
            evaluate(input, ej);
            input[j] = origJ;

            // Calculate the value of the Hessian at this index
            hessian[i][j] = (eij[outputIndex] - ei[outputIndex] -
                ej[outputIndex] + base[outputIndex]) / (EPSILON * EPSILON);
        }
    }
}

template <class T>
void Function<T>::calculateHessianParameters(const vector<T>& x,
        const size_t outputIndex, Dataset<T>& hessian)
{
    cout << "Function::calculateHessianParameters()" << endl;

    // Epsilon has to be set to a larger value than that used in calculating
    // the gradient because it will be squared in the calculations below. If it
    // is too small, we incur more significant rounding errors.
    const static T EPSILON = std::pow(std::numeric_limits<T>::epsilon(), T{0.25});
    const size_t N         = getNumParameters();
    const size_t M         = getOutputs();

    hessian.setSize(N, N);
    vector<T>& params = getParameters();

    // Create the temporary vectors we'll need
    static vector<T> base(M, T{});
    static vector<T> ei(M, T{});
    static vector<T> ej(M, T{});
    static vector<T> eij(M, T{});

    // Perform one evaluation with no changes to get a baseline measurement
    evaluate(x, base);

    // Using the method of finite differences, each element of the Hessian
    // can be approximated using the following formula:
    // H(i,j) = (f(x1,x2,...xi + h, ...xj + k...xn) - f(x1, x2 ,...xi + h...xn)
    //          - f(x1, x2, ... xj + k ... xn) + f(x1...xn)) / hk
    for (size_t i = 0; i < N; ++i)
    {
        // Modify i alone
        T origI      = params[i];
        params[i]   += EPSILON;
        evaluate(x, ei);
        params[i]    = origI;

        for (size_t j = 0; j < N; ++j)
        {
            // Modify i and j
            T origJ      = params[j];
            params[i]   += EPSILON;
            params[j]   += EPSILON;
            evaluate(x, eij);
            params[i]    = origI;
            params[j]    = origJ;

            // Modify j alone
            params[j] += EPSILON;
            evaluate(x, ej);
            params[j]  = origJ;

            // Calculate the value of the Hessian at this index
            hessian[i][j] = (eij[outputIndex] - ei[outputIndex] -
                ej[outputIndex] + base[outputIndex]) / (EPSILON * EPSILON);
        }
    }
}

// Initialize the parameters with random values from a normal distribution
// of the given mean and variance
template <class T>
void randomizeParameters(vector<T>& parameters,
    const T mean = T{}, const T variance = T{1.0})
{
    std::default_random_engine generator;
    std::normal_distribution<> rand(mean, variance);

    for (size_t i = 0; i < parameters.size(); ++i)
        parameters[i] = rand(generator);
}

};

#endif /* MODEL_H */

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
#include "Matrix.h"
#include "Error.h"

using std::cout;
using std::endl;
using std::vector;

// Hide the common method implementations in an anonymous namespace so they
// can't be seen outside of this file
namespace
{
    // Calculates the Jacobian matrix of the given function at the point 'x'
    // with respect to the parameters 'params' using the method of finite
    // differences. If params == x, the derivatives with respect to the inputs
    // are used. If params == the intrinsic parameters of the function, those
    // derivatives will be used.
    // NOTE: It is acceptable for 'x' and 'params' to be references to the same
    // vector. Although 'params' is not declared const, it is guaranteed to have
    // the same values after this function is called as it did before the
    // function was called.
    // NOTE 2: This method requires O(2*N) evaluations of the given function. If
    // there is a more efficient means for generating the Jacobian for a
    // particular function, it should be used instead.
    // NOTE 3: "Fn" is assumed to be a functor that has the methods getOutputs()
    // and evaluate(). The Function class defined below will work, but it's not
    // required.
    template <class T, template <class U> class Fn>
    void calculateJacobian(Fn<T>& function, const vector<T>& x,
        vector<T>& params, opkit::Matrix<T>& jacobian)
    {
        // Constants used in the finite differences approximation
        // Epsilon needs to be a reasonably small number, but the size depends on
        // the type (e.g. doubles need smaller values). We use the sqrt of the
        // machine epsilon as a good starting point.
        const static T EPSILON   = std::sqrt(std::numeric_limits<T>::epsilon());
        const static T INV_DENOM = T{0.5} / EPSILON;
        const size_t N           = params.size();
        const size_t M           = function.getOutputs();

        // Ensure the Jacobian matrix is large enough
        jacobian.resize(M, N);

        // Temporary vectors used for calculations
        static vector<T> derivativePrediction1(M, T{});
        static vector<T> derivativePrediction2(M, T{});

        // Eliminate the vector class [] overhead (very minor optimization).
        T* data = params.data();

        // The Jacobian is calculated one column at a time by changing one
        // parameter and measuring the effect on all M outputs.
        for (size_t p = 0; p < N; ++p)
        {
            // Save the original value of this parameter
            T orig = data[p];

            // Calculate the derivative of the function (y) with respect to
            // the current parameter, p, by slightly evaluating the function
            // at two points, one ahead and one behind p. This should yield
            // a better approximation of the derivative than only using one
            // point.
            data[p] += EPSILON;
            function.evaluate(x, derivativePrediction1);

            data[p] = orig - EPSILON;
            function.evaluate(x, derivativePrediction2);

            for (size_t r = 0; r < M; ++r)
            {
                jacobian(r,p) = INV_DENOM *
                    (derivativePrediction1[r] - derivativePrediction2[r]);
            }

            // Change the parameter back to its original value
            data[p] = orig;
        }
    }

    // Calculates the Hessian matrix of the given function at the point 'x' with
    // respect to the parameters 'params' using the method of finite differences
    // and with respect to the given output index. (The full Hessian for a
    // general N->M function is actually a 3rd order Tensor). If params == x,
    // the derivatives with respect to the inputs are used. If params == the
    // intrinsic parameters of the function, those derivatives will be used.
    // NOTE: It is acceptable for 'x' and 'params' to be references to the same
    // vector. Although 'params' is not declared const, it is guaranteed to have
    // the same values after this function is called as it did before the
    // function was called.
    // function. If there is a more efficient means for generating a Hessian for
    // a particular function, it should be used instead.
    // NOTE 3: "Fn" is assumed to be a functor that has the methods getOutputs()
    // and evaluate(). The Function class defined below will work, but it's not
    // required.
    template <class T, template <class U = T> class Fn>
    void calculateHessian(Fn<T>& function, const vector<T>& x,
        vector<T>& params, const size_t outputIndex, opkit::Matrix<T>& hessian)
    {
        // Epsilon has to be set to a larger value than that used in calculating
        // the gradient because it will be squared in the calculations below. If it
        // is too small, we incur more significant rounding errors.
        const static T EPSILON = std::pow(std::numeric_limits<T>::epsilon(), T{0.25});
        const static T DENOM   = T{4.0} * EPSILON * EPSILON;

        const size_t N = params.size();
        const size_t M = function.getOutputs();

        hessian.resize(N, N);
        T* data = params.data();

        // Create the temporary vectors we'll need
        static vector<T> plusplus(M, T{});
        static vector<T> plusminus(M, T{});
        static vector<T> minusplus(M, T{});
        static vector<T> minusminus(M, T{});

        // Using the method of finite differences, each element of the Hessian
        // can be approximated using the following formula:
        // H(i,j) = (f(x+h, y+h) - f(x+h, y-h) - f(x-h, y+h) + f(x-h, y-h)) / 4h^2
        for (size_t i = 0; i < N; ++i)
        {
            for (size_t j = 0; j < N; ++j)
            {
                T origI = data[i];
                T origJ = data[j];

                data[i] += EPSILON;
                data[j] += EPSILON;
                function.evaluate(x, plusplus);
                data[i] = origI;
                data[j] = origJ;

                data[i] += EPSILON;
                data[j] -= EPSILON;
                function.evaluate(x, plusminus);
                data[i] = origI;
                data[j] = origJ;

                data[i] -= EPSILON;
                data[j] += EPSILON;
                function.evaluate(x, minusplus);
                data[i] = origI;
                data[j] = origJ;

                data[i] -= EPSILON;
                data[j] -= EPSILON;
                function.evaluate(x, minusminus);
                data[i] = origI;
                data[j] = origJ;

                // Calculate the value of the Hessian at this index
                hessian(i, j) = (plusplus[outputIndex] - plusminus[outputIndex] -
                    minusplus[outputIndex] + minusminus[outputIndex]) / DENOM;
            }
        }
    }
}

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

    // Calculates the Jacobian of this function df(x)/dx with respect to the
    // function inputs. 'x' is the point at which the Jacobian should be
    // calculated, and the Jacobian itself is stored in 'Jacobian'.
    virtual void calculateJacobianInputs(const vector<T>& x, Matrix<T>& jacobian)
    {
        cout << "Function::calculateJacobianInputs()" << endl;
        ::calculateJacobian<T>(*this, x, (vector<T>&) x, jacobian);
    }

    // Calculates the Jacobian of this function df(x)/dx with respect to the
    // function parameters. 'x' is the point at which the Jacobian should be
    // calculated, and the Jacobian itself is stored in 'Jacobian'.
    virtual void calculateJacobianParameters(const vector<T>& x, Matrix<T>& jacobian)
    {
        cout << "Function::calculateJacobianParameters()" << endl;
        ::calculateJacobian<T>(*this, x, getParameters(), jacobian);
    }

    // Calculates the Hessian matrix of this function with respect to the
    // function inputs. 'x' is the point at which the Hessian should be
    // calculated, and the Hessian itself is stored in 'hessian'. Since an
    // arbitrary function can have many inputs and outputs, the second
    // derivative is technically a 3rd order tensor (3D matrix). This function
    // will only calculate the Hessian with respect to a single output, indexed
    // by 'outputIndex'. For scalar functions, 'outputIndex' should be 0.
    virtual void calculateHessianInputs(const vector<T>& x,
        const size_t outputIndex, Matrix<T>& hessian)
    {
        cout << "Function::calculateHessianInputs()" << endl;
        ::calculateHessian<T>(*this, x, (vector<T>&) x, outputIndex, hessian);
    }

    // Calculates the Hessian matrix of this function with respect to the
    // function parameters. 'x' is the point at which the Hessian should be
    // calculated, and the Hessian itself is stored in 'hessian'. Since an
    // arbitrary function can have many inputs and outputs, the second
    // derivative is technically a 3rd order tensor (3D matrix). This function
    // will only calculate the Hessian with respect to a single output, indexed
    // by 'outputIndex'. For scalar functions, 'outputIndex' should be 0.
    virtual void calculateHessianParameters(const vector<T>& x,
        const size_t outputIndex, Matrix<T>& hessian)
    {
        cout << "Function::calculateHessianParameters()" << endl;
        ::calculateHessian<T>(*this, x, getParameters(), outputIndex, hessian);
    }
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

// Initialize the parameters with random values from a normal distribution
// of the given mean and variance
template <class T>
void randomizeParameters(vector<T>& parameters,
    const double mean = T{}, const double variance = 1.0)
{
    std::default_random_engine generator;
    std::normal_distribution<> rand(mean, variance);

    for (size_t i = 0; i < parameters.size(); ++i)
        parameters[i] = rand(generator);
}

}

#endif /* MODEL_H */

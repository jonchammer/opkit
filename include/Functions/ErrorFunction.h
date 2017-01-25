/*
 * File:   ErrorFunction.h
 * Author: Jon C. Hammer
 *
 * Created on August 12, 2016, 9:58 AM
 */

#ifndef ERRORFUNCTION_H
#define ERRORFUNCTION_H

#include "Function.h"
#include "Dataset.h"
#include "Matrix.h"
#include "Acceleration.h"
#include "PrettyPrinter.h"

namespace opkit
{

template <class T, class Model>
class ErrorFunction
{
public:
    ErrorFunction(Model& baseFunction) :
        mBaseFunction(baseFunction)
    {
        // Do nothing
    }

    // Error functions compare the output of a base function on a given feature
    // to a known result. The interface presented in 'Function' for these
    // methods isn't really applicable to Error Functions, so these are
    // more intuitive replacements.
    virtual T evaluate(const Dataset<T>& features, const Dataset<T>& labels) = 0;

    // Note: The default implementations of the functions that calculate
    // derivatives with respect to the 'inputs' are quite slow and are not
    // particularly numerically stable. Those that calculate derivatives with
    // respect to the 'parameters' are much faster and more stable, but it would
    // still be a good idea for child classes to provide better implementations
    // of all of these functions if it is possible to do so.
    // --- Use the default implementations at your own risk. ---
    // NOTE: Gradients are averaged over each sample in the Dataset.
    virtual void calculateGradientInputs(const Dataset<T>& features,
        const Dataset<T>& labels, vector<T>& gradient);
    virtual void calculateGradientParameters(const Dataset<T>& features,
        const Dataset<T>& labels, vector<T>& gradient);
    virtual void calculateHessianInputs(const Dataset<T>& features,
        const Dataset<T>& labels, Matrix<T>& hessian);
    virtual void calculateHessianParameters(const Dataset<T>& features,
        const Dataset<T>& labels, Matrix<T>& hessian);

    // Returns the number of inputs to the function and the number of outputs,
    // respectively. Error functions only have 1 output.
    size_t getInputs()  const
    {
        return mBaseFunction.getInputs();
    }

    size_t getOutputs() const
    {
        return 1;
    }

    // Our 'parameters' are simply those of the base function. We forward the
    // calls wherever necessary.
    vector<T>& getParameters()
    {
        return mBaseFunction.getParameters();
    }

    const vector<T>& getParameters() const
    {
        return mBaseFunction.getParameters();
    }

    size_t getNumParameters() const
    {
        return mBaseFunction.getNumParameters();
    }

protected:
    Model& mBaseFunction;
};

template <class T, class Model>
void ErrorFunction<T, Model>::calculateGradientInputs(const Dataset<T>& features,
    const Dataset<T>& labels, vector<T>& gradient)
{
    cout << "ErrorFunction::calculateGradientInputs()" << endl;

    // Constants used in the finite differences approximation
    // Epsilon needs to be a reasonably small number, but the size depends on
    // the type (e.g. doubles need smaller values). We use the sqrt of the
    // machine epsilon as a good starting point.
    const static T EPSILON   = std::sqrt(std::numeric_limits<T>::epsilon());
    const static T DENOM     = T{2.0} * EPSILON;
    const size_t N           = getInputs();

    // Ensure the gradient vector is large enough
    std::fill(gradient.begin(), gradient.end(), T{});

    size_t rows = features.rows();
    for (size_t r = 0; r < rows; ++r)
    {
        // Yes, 'features' is declared const. We temporarily change one value in
        // one row, re-evaluate the function, and then revert the value to its
        // original state. The const-invariance of 'features' will therefore
        // be preserved.
        vector<T>& row = (vector<T>&) features[r];

        for (size_t p = 0; p < N; ++p)
        {
            // Save the original value of this input
            T orig = row[p];

            // Start by evaluating the function for a point slightly ahead
            row[p] += EPSILON;
            T y     = evaluate(features, labels);

            // Calculate the derivative of the function (y) with respect to
            // the current parameter, p, by slightly changing that parameter
            // and measuring comparing the output that with no change applied.
            row[p] = orig - EPSILON;
            T y2   = evaluate(features, labels);
            row[p] = orig;

            gradient[p] += ((y - y2) / DENOM);
        }
    }

    // Calculate the average gradient for the batch
    vScale(gradient.data(), T{1.0/rows}, N);
}

template <class T, class Model>
void ErrorFunction<T, Model>::calculateGradientParameters(
    const Dataset<T>& features, const Dataset<T>& labels, vector<T>& gradient)
{
    cout << "ErrorFunction::calculateGradientParameters()" << endl;

    // Constants used in the finite differences approximation
    // Epsilon needs to be a reasonably small number, but the size depends on
    // the type (e.g. doubles need smaller values). We use the sqrt of the
    // machine epsilon as a good starting point.
    const static T EPSILON = std::sqrt(std::numeric_limits<T>::epsilon());
    const static T DENOM   = features.rows() * T{2.0} * EPSILON;
    const size_t N         = getNumParameters();

     // Start by evaluating the function without any modifications
    T* parameters = getParameters().data();

    for (size_t p = 0; p < N; ++p)
    {
        // Save the original value of this parameter
        T orig = parameters[p];

        // Start by evaluating the function for a point slightly ahead
        parameters[p] += EPSILON;
        T y            = evaluate(features, labels);

        // Calculate the derivative of the function (y) with respect to
        // the current parameter, p, by slightly changing that parameter
        // and measuring comparing the output that with no change applied.
        parameters[p] = orig - EPSILON;
        T y2          = evaluate(features, labels);
        parameters[p] = orig;

        // Divide by the number of rows to get the average gradient
        gradient[p] = (y - y2) / DENOM;
    }
}

template <class T, class Model>
void ErrorFunction<T, Model>::calculateHessianInputs(const Dataset<T>& features,
    const Dataset<T>& labels, Matrix<T>& hessian)
{
    cout << "ErrorFunction::calculateHessianInputs()" << endl;

    // Epsilon has to be set to a larger value than that used in calculating
    // the gradient because it will be squared in the calculations below. If it
    // is too small, we incur more significant rounding errors.
    const static T EPSILON = std::pow(std::numeric_limits<T>::epsilon(), T{0.25});
    const static T DENOM   = T{4.0} * EPSILON * EPSILON;
    const size_t N         = mBaseFunction.getInputs();

    hessian.resize(N, N);

    // Using the method of finite differences, each element of the Hessian
    // can be approximated using the following formula:
    // H(i,j) = (f(x+h, y+h) - f(x+h, y-h) - f(x-h, y+h) + f(x-h, y-h)) / 4h^2
    for (size_t k = 0; k < features.rows(); ++k)
    {
        // Yes, 'features' is declared const. We temporarily change one value in
        // one row, re-evaluate the function, and then revert the value to its
        // original state. The const-invariance of 'features' will therefore
        // be preserved.
        vector<T>& row = (vector<T>&) features.row(k);

        for (size_t i = 0; i < N; ++i)
        {
            for (size_t j = 0; j < N; ++j)
            {
                T origI = row[i];
                T origJ = row[j];

                row[i]    += EPSILON;
                row[j]    += EPSILON;
                T plusplus = evaluate(features, labels);
                row[i]     = origI;
                row[j]     = origJ;

                row[i]     += EPSILON;
                row[j]     -= EPSILON;
                T plusminus = evaluate(features, labels);
                row[i]      = origI;
                row[j]      = origJ;

                row[i]     -= EPSILON;
                row[j]     += EPSILON;
                T minusplus = evaluate(features, labels);
                row[i]      = origI;
                row[j]      = origJ;

                row[i]      -= EPSILON;
                row[j]      -= EPSILON;
                T minusminus = evaluate(features, labels);
                row[i]       = origI;
                row[j]       = origJ;

                // Calculate the value of the Hessian at this index
                hessian(i, j) += ((plusplus - plusminus - minusplus + minusminus) / DENOM);
            }
        }
    }
}

template <class T, class Model>
void ErrorFunction<T, Model>::calculateHessianParameters(
    const Dataset<T>& features, const Dataset<T>& labels, Matrix<T>& hessian)
{
    cout << "ErrorFunction::calculateHessianParameters()" << endl;

    // Epsilon has to be set to a larger value than that used in calculating
    // the gradient because it will be squared in the calculations below. If it
    // is too small, we incur more significant rounding errors.
    const static T EPSILON = std::pow(std::numeric_limits<T>::epsilon(), T{0.25});
    const static T DENOM   = T{4.0} * EPSILON * EPSILON;
    const size_t N         = mBaseFunction.getNumParameters();

    hessian.resize(N, N);
    T* params = getParameters().data();

    // Using the method of finite differences, each element of the Hessian
    // can be approximated using the following formula:
    // H(i,j) = (f(x+h, y+h) - f(x+h, y-h) - f(x-h, y+h) + f(x-h, y-h)) / 4h^2
    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            T origI = params[i];
            T origJ = params[j];

            params[i] += EPSILON;
            params[j] += EPSILON;
            T plusplus = evaluate(features, labels);
            params[i]  = origI;
            params[j]  = origJ;

            params[i]  += EPSILON;
            params[j]  -= EPSILON;
            T plusminus = evaluate(features, labels);
            params[i]   = origI;
            params[j]   = origJ;

            params[i]  -= EPSILON;
            params[j]  += EPSILON;
            T minusplus = evaluate(features, labels);
            params[i]   = origI;
            params[j]   = origJ;

            params[i]   -= EPSILON;
            params[j]   -= EPSILON;
            T minusminus = evaluate(features, labels);
            params[i]    = origI;
            params[j]    = origJ;

            // Calculate the value of the Hessian at this index
            hessian(i, j) += ((plusplus - plusminus - minusplus + minusminus) / DENOM);
        }
    }
}

};
#endif /* ERRORFUNCTION_H */

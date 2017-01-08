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

namespace opkit
{

template <class T, class Model>
class ErrorFunction
{
public:
    ErrorFunction(Model& baseFunction);

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
        const Dataset<T>& labels, Dataset<T>& hessian);
    virtual void calculateHessianParameters(const Dataset<T>& features,
        const Dataset<T>& labels, Dataset<T>& hessian);

    // Returns the number of inputs to the function and the number of outputs,
    // respectively. Error functions only have 1 output.
    virtual size_t getInputs()  const;
    virtual size_t getOutputs() const;

    // Our 'parameters' are simply those of the base function. We forward the
    // calls wherever necessary.
    virtual vector<T>& getParameters();
    virtual const vector<T>& getParameters() const;
    virtual size_t getNumParameters() const;

protected:
    Model& mBaseFunction;
};

template <class T, class Model>
ErrorFunction<T, Model>::ErrorFunction(Model& baseFunction) :
    mBaseFunction(baseFunction)
{
    // Do nothing
}

template <class T, class Model>
size_t ErrorFunction<T, Model>::getInputs()  const
{
    return mBaseFunction.getInputs();
}

template <class T, class Model>
size_t ErrorFunction<T, Model>::getOutputs() const
{
    return 1;
}

template <class T, class Model>
vector<T>& ErrorFunction<T, Model>::getParameters()
{
    return mBaseFunction.getParameters();
}

template <class T, class Model>
const vector<T>& ErrorFunction<T, Model>::getParameters() const
{
    return mBaseFunction.getParameters();
}

template <class T, class Model>
size_t ErrorFunction<T, Model>::getNumParameters() const
{
    return mBaseFunction.getNumParameters();
}

template <class T, class Model>
void ErrorFunction<T, Model>::calculateGradientInputs(const Dataset<T>& features,
    const Dataset<T>& labels, vector<T>& gradient)
{
    cout << "ErrorFunction::calculateGradientInputs()" << endl;

    // Constants used in the finite differences approximation
    // Epsilon needs to be a reasonably small number, but the size depends on
    // the type (e.g. doubles need smaller values). We use the sqrt of the
    // machine epsilon as a good starting point.
    const static T EPSILON = std::sqrt(std::numeric_limits<T>::epsilon());
    const size_t N         = getInputs();

    // Ensure the gradient vector is large enough
    std::fill(gradient.begin(), gradient.end(), T{});

     // Start by evaluating the function without any modifications
    T y = evaluate(features, labels);

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

            // Calculate the derivative of the function (y) with respect to
            // the current parameter, p, by slightly changing that parameter
            // and measuring comparing the output that with no change applied.
            row[p] += EPSILON;
            T y2 = evaluate(features, labels);

            gradient[p] += ((y2 - y) / EPSILON);

            // Change the parameter back to its original value
            row[p] = orig;
        }
    }

    // Calculate the average gradient for the batch
    for (size_t i = 0; i < N; ++i)
        gradient[i] /= rows;
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
    const size_t N         = getNumParameters();

     // Start by evaluating the function without any modifications
    vector<T>& parameters = getParameters();
    T y = evaluate(features, labels);

    for (size_t p = 0; p < N; ++p)
    {
        // Save the original value of this parameter
        T orig = parameters[p];

        // Calculate the derivative of the function (y) with respect to
        // the current parameter, p, by slightly changing that parameter
        // and measuring comparing the output that with no change applied.
        parameters[p] += EPSILON;
        T y2 = evaluate(features, labels);

        // Divide by the number of rows to get the average gradient
        gradient[p] = (y2 - y) / (EPSILON * features.rows());

        // Change the parameter back to its original value
        parameters[p] = orig;
    }
}

template <class T, class Model>
void ErrorFunction<T, Model>::calculateHessianInputs(const Dataset<T>& features,
    const Dataset<T>& labels, Dataset<T>& hessian)
{
    cout << "ErrorFunction::calculateHessianInputs()" << endl;

    // Epsilon has to be set to a larger value than that used in calculating
    // the gradient because it will be squared in the calculations below. If it
    // is too small, we incur more significant rounding errors.
    const static T EPSILON = std::pow(std::numeric_limits<T>::epsilon(), T{0.25});
    const size_t N         = mBaseFunction.getInputs();

    hessian.setSize(N, N);
    hessian.setAll(T{});

    // Perform one evaluation with no changes to get a baseline measurement
    T base = evaluate(features, labels);

    // Using the method of finite differences, each element of the Hessian
    // can be approximated using the following formula:
    // H(i,j) = (f(x1,x2,...xi + h, ...xj + k...xn) - f(x1, x2 ,...xi + h...xn)
    //- f(x1, x2, ... xj + k ... xn) + f(x1...xn)) / hk
    for (size_t k = 0; k < features.rows(); ++k)
    {
        // Yes, 'features' is declared const. We temporarily change one value in
        // one row, re-evaluate the function, and then revert the value to its
        // original state. The const-invariance of 'features' will therefore
        // be preserved.
        vector<T>& row = (vector<T>&) features.row(k);

        for (size_t i = 0; i < N; ++i)
        {
            // Modify i alone
            T origI = row[i];
            row[i] += EPSILON;
            T ei    = evaluate(features, labels);
            row[i]  = origI;

            for (size_t j = 0; j < N; ++j)
            {
                // Modify i and j
                T origJ = row[j];
                row[i] += EPSILON;
                row[j] += EPSILON;
                T eij   = evaluate(features, labels);
                row[i]  = origI;
                row[j]  = origJ;

                // Modify j alone
                row[j]+= EPSILON;
                T ej   = evaluate(features, labels);
                row[j] = origJ;

                // Calculate the value of the Hessian at this index
                hessian[i][j] = (eij - ei - ej + base) / (EPSILON * EPSILON);
            }
        }
    }
}

template <class T, class Model>
void ErrorFunction<T, Model>::calculateHessianParameters(
    const Dataset<T>& features, const Dataset<T>& labels, Dataset<T>& hessian)
{
    cout << "ErrorFunction::calculateHessianParameters()" << endl;

    // Epsilon has to be set to a larger value than that used in calculating
    // the gradient because it will be squared in the calculations below. If it
    // is too small, we incur more significant rounding errors.
    const static T EPSILON = std::pow(std::numeric_limits<T>::epsilon(), T{0.25});
    const size_t N         = mBaseFunction.getNumParameters();

    hessian.setSize(N, N);
    vector<T>& params = getParameters();

    // Perform one evaluation with no changes to get a baseline measurement
    T base = evaluate(features, labels);

    // Using the method of finite differences, each element of the Hessian
    // can be approximated using the following formula:
    // H(i,j) = (f(x1,x2,...xi + h, ...xj + k...xn) - f(x1, x2 ,...xi + h...xn)
    //- f(x1, x2, ... xj + k ... xn) + f(x1...xn)) / hk
    for (size_t i = 0; i < N; ++i)
    {
        // Modify i alone
        T origI    = params[i];
        params[i] += EPSILON;
        T ei       = evaluate(features, labels);
        params[i]  = origI;

        for (size_t j = 0; j < N; ++j)
        {
            // Modify i and j
            T origJ      = params[j];
            params[i]   += EPSILON;
            params[j]   += EPSILON;
            T eij        = evaluate(features, labels);
            params[i]    = origI;
            params[j]    = origJ;

            // Modify j alone
            params[j] += EPSILON;
            T ej       = evaluate(features, labels);
            params[j]  = origJ;

            // Calculate the value of the Hessian at this index
            hessian[i][j] = (eij - ei - ej + base) / (EPSILON * EPSILON);
        }
    }
}

};
#endif /* ERRORFUNCTION_H */

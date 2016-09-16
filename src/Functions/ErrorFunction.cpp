#include "ErrorFunction.h"

ErrorFunction::ErrorFunction(Function& baseFunction) : mBaseFunction(baseFunction)
{
    // Do nothing
}

size_t ErrorFunction::getInputs()  const 
{
    return mBaseFunction.getInputs();
}

size_t ErrorFunction::getOutputs() const 
{
    return 1;
}

vector<double>& ErrorFunction::getParameters()             
{ 
    return mBaseFunction.getParameters(); 
}

const vector<double>& ErrorFunction::getParameters() const 
{ 
    return mBaseFunction.getParameters(); 
}

size_t ErrorFunction::getNumParameters() const             
{ 
    return mBaseFunction.getNumParameters(); 
}

void ErrorFunction::calculateGradientInputs(const Matrix& features, 
    const Matrix& labels, vector<double>& gradient)
{
    cout << "ErrorFunction::calculateGradientInputs()" << endl;
    
    // Constants used in the finite differences approximation
    const double EPSILON = 1.0E-10;
    const size_t N       = getInputs();
    
    // Ensure the gradient vector is large enough
    gradient.resize(N);
    std::fill(gradient.begin(), gradient.end(), 0.0);
    
     // Start by evaluating the function without any modifications
    double y = evaluate(features, labels);

    for (size_t r = 0; r < features.rows(); ++r)
    {
        // Yes, 'features' is declared const. We temporarily change one value in
        // one row, re-evaluate the function, and then revert the value to its
        // original state. The const-invariance of 'features' will therefore
        // be preserved.
        vector<double>& row = (vector<double>&) features[r];
        
        for (size_t p = 0; p < N; ++p)
        {
            // Save the original value of this input
            double orig = row[p];

            // Calculate the derivative of the function (y) with respect to
            // the current parameter, p, by slightly changing that parameter
            // and measuring comparing the output that with no change applied.
            row[p] += EPSILON;
            double y2 = evaluate(features, labels);

            gradient[p] += (y2 - y) / EPSILON;

            // Change the parameter back to its original value
            row[p] = orig;
        }
    }
}

void ErrorFunction::calculateGradientParameters(const Matrix& features, 
    const Matrix& labels, vector<double>& gradient)
{
    cout << "ErrorFunction::calculateGradientParameters()" << endl;
    
    // Constants used in the finite differences approximation
    const double EPSILON = 1.0E-10;
    const size_t N       = getNumParameters();
    
    // Ensure the gradient vector is large enough
    gradient.resize(N);
    
     // Start by evaluating the function without any modifications
    vector<double>& parameters = getParameters();
    double y = evaluate(features, labels);

    for (size_t p = 0; p < N; ++p)
    {
        // Save the original value of this parameter
        double orig = parameters[p];

        // Calculate the derivative of the function (y) with respect to
        // the current parameter, p, by slightly changing that parameter
        // and measuring comparing the output that with no change applied.
        parameters[p] += EPSILON;
        double y2 = evaluate(features, labels);

        gradient[p] = (y2 - y) / EPSILON;
        
        // Change the parameter back to its original value
        parameters[p] = orig;
    }
}
    
void ErrorFunction::calculateHessianInputs(const Matrix& features,
    const Matrix& labels, Matrix& hessian)
{
    cout << "ErrorFunction::calculateHessianInputs()" << endl;
    
    // Epsilon has to be set to a larger value than that used in calculating
    // the gradient because it will be squared in the calculations below. If it
    // is too small, we incur more significant rounding errors.
    const double EPSILON = 1E-4;
    const size_t N       = mBaseFunction.getInputs();
    
    hessian.setSize(N, N);
    hessian.setAll(0.0);

    // Perform one evaluation with no changes to get a baseline measurement
    double base = evaluate(features, labels);

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
        vector<double>& row = (vector<double>&) features.row(k);
        
        for (size_t i = 0; i < N; ++i)
        {
            // Modify i alone
            double origI = row[i];
            row[i]      += EPSILON;
            double ei    = evaluate(features, labels);
            row[i]       = origI;

            for (size_t j = 0; j < N; ++j)
            {
                // Modify i and j
                double origJ = row[j];
                row[i]      += EPSILON;
                row[j]      += EPSILON;
                double eij   = evaluate(features, labels);
                row[i]       = origI;
                row[j]       = origJ;

                // Modify j alone
                row[j]   += EPSILON;
                double ej = evaluate(features, labels);
                row[j]    = origJ;

                // Calculate the value of the Hessian at this index
                hessian[i][j] = (eij - ei - ej + base) / (EPSILON * EPSILON);
            }
        }
    }
}

void ErrorFunction::calculateHessianParameters(const Matrix& features, 
    const Matrix& labels, Matrix& hessian)
{
    cout << "ErrorFunction::calculateHessianParameters()" << endl;
    
    // Epsilon has to be set to a larger value than that used in calculating
    // the gradient because it will be squared in the calculations below. If it
    // is too small, we incur more significant rounding errors.
    const double EPSILON = 1E-4;
    const size_t N       = mBaseFunction.getNumParameters();
    
    hessian.setSize(N, N);
    vector<double>& params = getParameters();
    
    // Perform one evaluation with no changes to get a baseline measurement
    double base = evaluate(features, labels);

    // Using the method of finite differences, each element of the Hessian
    // can be approximated using the following formula:
    // H(i,j) = (f(x1,x2,...xi + h, ...xj + k...xn) - f(x1, x2 ,...xi + h...xn) 
    //- f(x1, x2, ... xj + k ... xn) + f(x1...xn)) / hk
    for (size_t i = 0; i < N; ++i)
    {
        // Modify i alone
        double origI = params[i];
        params[i]   += EPSILON;
        double ei    = evaluate(features, labels);
        params[i]    = origI;
        
        for (size_t j = 0; j < N; ++j)
        {
            // Modify i and j
            double origJ = params[j];
            params[i]   += EPSILON;
            params[j]   += EPSILON;
            double eij   = evaluate(features, labels);
            params[i]    = origI;
            params[j]    = origJ;
            
            // Modify j alone
            params[j] += EPSILON;
            double ej  = evaluate(features, labels);
            params[j]  = origJ;
            
            // Calculate the value of the Hessian at this index
            hessian[i][j] = (eij - ei - ej + base) / (EPSILON * EPSILON);
        }
    }
}
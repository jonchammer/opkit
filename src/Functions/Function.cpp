/* 
 * File:   Model.cpp
 * Author: Jon C. Hammer
 * 
 * Created on July 9, 2016, 7:56 PM
 */

#include "Function.h"
#include "PrettyPrinter.h"

void Function::calculateJacobianInputs(const vector<double>& x, 
        Matrix& jacobian)
{
    cout << "Function::calculateJacobianInputs()" << endl;

    // Constants used in the finite differences approximation
    const double EPSILON = 1.0E-10;
    const size_t N       = getInputs();
    const size_t M       = getOutputs();
    
    // Ensure the Jacobian matrix is large enough
    jacobian.setSize(M, N);
    
    // Temporary vectors used for calculations
    static vector<double> prediction(M, 0.0);
    static vector<double> derivativePrediction(M, 0.0);
    static vector<double> input(N, 0.0);
    
    // Start by evaluating the function without any modifications
    std::copy(x.begin(), x.end(), input.begin());
    evaluate(input, prediction);

    // The Jacobian is calculated one column at a time by changing one input
    // and measuring the effect on all M outputs.
    for (size_t p = 0; p < N; ++p)
    {
        // Save the original value of this input
        double orig = input[p];

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

void Function::calculateJacobianParameters(const vector<double>& x,
        Matrix& jacobian)
{
    cout << "Function::calculateJacobianParameters()" << endl;
    
    // Constants used in the finite differences approximation
    const double EPSILON = 1.0E-10;
    const size_t N       = getNumParameters();
    const size_t M       = getOutputs();
    
    // Ensure the Jacobian matrix is large enough
    jacobian.setSize(M, N);
    
    // Temporary vectors used for calculations
    static vector<double> prediction(M, 0.0);
    static vector<double> derivativePrediction(M, 0.0);
    
    // Start by evaluating the function without any modifications
    vector<double>& parameters = getParameters();
    evaluate(x, prediction);

    for (size_t p = 0; p < N; ++p)
    {
        // Save the original value of this parameter
        double orig = parameters[p];

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
    
void Function::calculateHessianInputs(const vector<double>& x,
        const size_t outputIndex, Matrix& hessian)
{
    cout << "Function::calculateHessianInputs()" << endl;
    
    // Epsilon has to be set to a larger value than that used in calculating
    // the gradient because it will be squared in the calculations below. If it
    // is too small, we incur more significant rounding errors.
    const double EPSILON = 1E-4;
    const size_t N       = getInputs();
    const size_t M       = getOutputs();
    hessian.setSize(N, N);

    // Create the temporary vectors we'll need
    static vector<double> base(M, 0.0);
    static vector<double> ei(M, 0.0);
    static vector<double> ej(M, 0.0);
    static vector<double> eij(M, 0.0);
    static vector<double> input(N, 0.0);
    
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
        double origI = input[i];
        input[i]    += EPSILON;
        evaluate(input, ei);
        input[i]     = origI;
        
        for (size_t j = 0; j < N; ++j)
        {
            // Modify i and j
            double origJ = input[j];
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

void Function::calculateHessianParameters(const vector<double>& x,
        const size_t outputIndex, Matrix& hessian)
{
    cout << "Function::calculateHessianParameters()" << endl;
    
    // Epsilon has to be set to a larger value than that used in calculating
    // the gradient because it will be squared in the calculations below. If it
    // is too small, we incur more significant rounding errors.
    const double EPSILON = 1E-4;
    const size_t N       = getNumParameters();
    const size_t M       = getOutputs();
    
    hessian.setSize(N, N);
    vector<double>& params = getParameters();
    
    // Create the temporary vectors we'll need
    static vector<double> base(M, 0.0);
    static vector<double> ei(M, 0.0);
    static vector<double> ej(M, 0.0);
    static vector<double> eij(M, 0.0);
    
    // Perform one evaluation with no changes to get a baseline measurement
    evaluate(x, base);
    
    // Using the method of finite differences, each element of the Hessian
    // can be approximated using the following formula:
    // H(i,j) = (f(x1,x2,...xi + h, ...xj + k...xn) - f(x1, x2 ,...xi + h...xn) 
    //          - f(x1, x2, ... xj + k ... xn) + f(x1...xn)) / hk
    for (size_t i = 0; i < N; ++i)
    {
        // Modify i alone
        double origI = params[i];
        params[i]   += EPSILON;
        evaluate(x, ei);
        params[i]    = origI;
        
        for (size_t j = 0; j < N; ++j)
        {
            // Modify i and j
            double origJ = params[j];
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

void randomizeParameters(vector<double>& parameters, 
    const double mean, const double variance)
{
    std::default_random_engine generator;
	std::normal_distribution<> rand(mean, variance);
    
    for (size_t i = 0; i < parameters.size(); ++i)
        parameters[i] = rand(generator);
}

//Model::Model()
//{
//    // Do nothing
//}
//
//Model::Model(int numParameters) : mParameters(numParameters, 0.0)
//{
//    // Do nothing
//}
//
//Model::Model(const Model& orig)
//    : mParameters(orig.mParameters)
//{
//    // Do nothing
//}
//
//void Model::calculateGradient(const vector<double>& feature, const vector<double>& label, vector<double>& gradient)
//{
//    const double EPSILON = 1.0E-10;
//    
//    // Set the gradient to the zero vector
//    gradient.resize(mParameters.size());
//    std::fill(gradient.begin(), gradient.end(), 0.0);
//    
//    // Temporary vectors used for calculations
//    static vector<double> prediction(label.size(), 0.0);
//    static vector<double> derivativePrediction(label.size(), 0.0);
//    
//    // dSSE                                     d
//    // ----  = sum(-2 * (label - prediction) * --- (prediction))
//    // dPi                                     dPi
//    //
//    // For multivariate data, there are more terms inside the sum,
//    // one for each output
//    //                                       d
//    // qi = -2 * (label_i - prediction_i) * --- (prediction_i)
//    //                                      dPi
//    // dSSE
//    // ---- = sum(q1 + q2 + ...)
//    //  dPi
//    evaluate(feature, prediction);
//
//    for (size_t p = 0; p < mParameters.size(); ++p)
//    {
//        // Save the original value of this parameter
//        double orig = mParameters[p];
//
//        // Calculate the derivative of the function (y) with respect to
//        // the current parameter, p, by slightly changing the value
//        // and measuring comparing the output to the original output
//        mParameters[p] += EPSILON;
//        evaluate(feature, derivativePrediction);
//
//        // Contributions for each output are summed together
//        for (size_t j = 0; j < label.size(); ++j)
//        {
//            double dydp  = (derivativePrediction[j] - prediction[j]) / EPSILON;
//            gradient[p] += -2.0 * (label[j] - prediction[j]) * dydp;
//        }
//
//        // Change the parameter back to its original value
//        mParameters[p] = orig;
//    }
//}
//
//void Model::calculateGradient(const Matrix& features, const Matrix& labels, vector<double>& gradient)
//{
//    // Set the gradient to the zero vector
//    gradient.resize(mParameters.size());
//    std::fill(gradient.begin(), gradient.end(), 0.0);
//    
//    // Temporary storage for intermediate gradient values
//    static vector<double> intermediates;
//    
//    for (size_t i = 0; i < features.rows(); ++i)
//    {
//        // Get the individual gradient for this pair
//        calculateGradient(features[i], labels[i], intermediates);
//        
//        // Add the results to the running total
//        std::transform(gradient.begin(), gradient.end(), intermediates.begin(), 
//            gradient.begin(), std::plus<double>());
//    }
//}
//
//double Model::measureSSE(const Matrix& features, const Matrix& labels)
//{
//    // Initialize variables
//    double sse = 0.0;
//    vector<double> prediction(labels.cols(), 0.0);
//    
//    // Calculate the SSE
//    for (size_t i = 0; i < features.rows(); ++i)
//    {
//        evaluate(features[i], prediction);
//                
//        const vector<double>& row = labels[i];
//        for (size_t j = 0; j < labels.cols(); ++j)
//        {
//            double d = row[j] - prediction[j];
//            
//            // For categorical columns, use Hamming distance instead
//            if (d != 0.0 && labels.valueCount(j) > 0)
//                d = 1.0;
//
//            sse += (d * d);
//        }
//    }
//
//    return sse;
//}
//
//void Model::randomizeParameters(double mean, double variance)
//{
//    std::default_random_engine generator;
//	std::normal_distribution<> rand(mean, variance);
//    
//    for (size_t i = 0; i < mParameters.size(); ++i)
//        mParameters[i] = rand(generator);
//}

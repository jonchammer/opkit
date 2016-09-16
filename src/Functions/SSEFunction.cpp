#include "SSEFunction.h"
#include "PrettyPrinter.h"

SSEFunction::SSEFunction(Function& baseFunction)
    : ErrorFunction(baseFunction)
{
    // Do nothing
}

double SSEFunction::evaluate(const Matrix& features, const Matrix& labels)
{
    // Initialize variables
    double sum = 0.0;
    static vector<double> prediction(labels.cols(), 0.0);
    prediction.resize(labels.cols());
    
    // Calculate the SSE
    for (size_t i = 0; i < features.rows(); ++i)
    {
        mBaseFunction.evaluate(features[i], prediction);
                
        const vector<double>& row = labels[i];
        for (size_t j = 0; j < labels.cols(); ++j)
        {
            double d = row[j] - prediction[j];
            
            // For categorical columns, use Hamming distance instead
            //if (d != 0.0 && labels.valueCount(j) > 0)
            //    d = 1.0;

            sum += (d * d);
        }
    }
    
    return sum;
}

void SSEFunction::calculateGradientInputs(const Matrix& features, 
    const Matrix& labels, vector<double>& gradient)
{
    // When SSE is the error function, the gradient is simply the error vector
    // multiplied by the model's Jacobian.
    const size_t N = mBaseFunction.getInputs();
    const size_t M = mBaseFunction.getOutputs();
    
    // Set the gradient to the zero vector
    gradient.resize(N);
    std::fill(gradient.begin(), gradient.end(), 0.0);

    static Matrix baseJacobian;
    static vector<double> evaluation(M);
    static vector<double> error(M);
    evaluation.resize(M);
    error.resize(M);
    
    for (size_t i = 0; i < labels.rows(); ++i)
    {
        // Calculate the Jacobian matrix of the base function at this point with
        // respect to the inputs
        mBaseFunction.calculateJacobianInputs(features[i], baseJacobian);
  
        // Calculate the error for this sample
        if (mBaseFunction.cachesLastEvaluation())
            mBaseFunction.getLastEvaluation(evaluation);
        else mBaseFunction.evaluate(features[i], evaluation);
        
        for (size_t j = 0; j < M; ++j)
            error[j] = labels[i][j] - evaluation[j];
               
        for (size_t j = 0; j < N; ++j)
        {
            // Multiply the error by the model's Jacobian,
            double sum = 0.0;
            for (size_t k = 0; k < M; ++k)
                sum += error[k] * baseJacobian[k][j];
            
            // Add the result to the running total for the gradient
            gradient[j] += -2.0 * sum;
        }
    }
}

void SSEFunction::calculateGradientParameters(const Matrix& features, 
    const Matrix& labels, vector<double>& gradient)
{
    // When SSE is the error function, the gradient is simply the error vector
    // multiplied by the model's Jacobian.
    const size_t N = mBaseFunction.getNumParameters();
    const size_t M = mBaseFunction.getOutputs();
    
    // Set the gradient to the zero vector
    gradient.resize(N);
    std::fill(gradient.begin(), gradient.end(), 0.0);
    
    static Matrix baseJacobian;
    static vector<double> evaluation(M);
    static vector<double> error(M);
    evaluation.resize(M);
    error.resize(M);
    
    for (size_t i = 0; i < labels.rows(); ++i)
    {
        // Calculate the Jacobian matrix of the base function at this point with
        // respect to the model parameters
        mBaseFunction.calculateJacobianParameters(features[i], baseJacobian);
        
        // Calculate the error for this sample
        if (mBaseFunction.cachesLastEvaluation())
            mBaseFunction.getLastEvaluation(evaluation);
        else mBaseFunction.evaluate(features[i], evaluation);
        
        for (size_t j = 0; j < M; ++j)
            error[j] = labels[i][j] - evaluation[j];
               
        for (size_t j = 0; j < N; ++j)
        {
            // Multiply the error by the model's Jacobian,
            double sum = 0.0;
            for (size_t k = 0; k < M; ++k)
                sum += error[k] * baseJacobian[k][j];
            
            // Add the result to the running total for the gradient
            gradient[j] += -2.0 * sum;
        }
    }
}

void SSEFunction::calculateHessianInputs(const Matrix& features,
    const Matrix& labels, Matrix& hessian)
{
    // When SSE is the error function, it is better to calculate the Hessian
    // directly using the following formula than to use finite differences.
    //
    // H = 2 * ((J^T * J) - sum_i((y_i - f(x_i, theta)) * H_i)), where
    //
    // H_i is the model's Hessian matrix with respect to output i
    // J is the model's Jacobian matrix
    
    const size_t N = mBaseFunction.getInputs();
    const size_t M = mBaseFunction.getOutputs();
    
    // Declare the temporary variables we'll need
    Matrix jacobian;
    Matrix jacobianSquare;
    Matrix localHessian;
    Matrix sumOfLocalHessians;
    vector<double> evaluation(M);
    vector<double> error(M);
    
    hessian.setSize(N, N);
    jacobian.setSize(M, N);
    jacobianSquare.setSize(N, N);
    localHessian.setSize(N, N);
    sumOfLocalHessians.setSize(N, N);
    
    hessian.setAll(0.0);
    
    for(size_t i = 0; i < features.rows(); ++i)
    {
        // Calculate the Jacobian matrix of the base function at this point with
        // respect to the model parameters
        mBaseFunction.calculateJacobianInputs(features[i], jacobian);

        // Calculate the square of the Jacobian matrix. 
        // TODO: Calculate J^T and work with that for better cache performance
        // c1 -> r1, c2 -> r2, r -> c. Reverse the indices
        for (size_t c1 = 0; c1 < N; ++c1)
        {
            for (size_t c2 = 0; c2 < N; ++c2)
            {
                double sum = 0.0;
                for (size_t r = 0; r < M; ++r)
                    sum += jacobian[r][c1] * jacobian[r][c2];

                jacobianSquare[c1][c2] = sum;
            }
        }
        
        // Calculate the error for this sample
        if (mBaseFunction.cachesLastEvaluation())
            mBaseFunction.getLastEvaluation(evaluation);
        else mBaseFunction.evaluate(features[i], evaluation);
        
        for (size_t j = 0; j < M; ++j)
            error[j] = labels[i][j] - evaluation[j];
        
        // Calculate the sum of the local Hessians
        sumOfLocalHessians.setAll(0.0);
        for (size_t j = 0; j < M; ++j)
        {
            // Calculate the local Hessian for output j
            mBaseFunction.calculateHessianInputs(features[i], j, localHessian);
            
            // Multiply by the error constant and add to the running total
            for (size_t r = 0; r < N; ++r)
            {
                for (size_t c = 0; c < N; ++c)
                    sumOfLocalHessians[r][c] += error[j] * localHessian[r][c];
            }
        }
        
        // Finally, calculate the Hessian
        for (size_t r = 0; r < N; ++r)
        {
            for (size_t c = 0; c < N; ++c)
            {
                hessian[r][c] += 2.0 * 
                    (jacobianSquare[r][c] - sumOfLocalHessians[r][c]);
            }
        }
    }
}

void SSEFunction::calculateHessianParameters(const Matrix& features, 
    const Matrix& labels, Matrix& hessian)
{
    // When SSE is the error function, it is better to calculate the Hessian
    // directly using the following formula than to use finite differences.
    //
    // H = 2 * ((J^T * J) - sum_i((y_i - f(x_i, theta)) * H_i)), where
    //
    // H_i is the model's Hessian matrix with respect to output i
    // J is the model's Jacobian matrix
    
    const size_t N = mBaseFunction.getNumParameters();
    const size_t M = mBaseFunction.getOutputs();
    
    // Declare the temporary variables we'll need
    Matrix jacobian;
    Matrix jacobianSquare;
    Matrix localHessian;
    Matrix sumOfLocalHessians;
    vector<double> evaluation(M);
    vector<double> error(M);
    
    hessian.setSize(N, N);
    jacobian.setSize(M, N);
    jacobianSquare.setSize(N, N);
    localHessian.setSize(N, N);
    sumOfLocalHessians.setSize(N, N);
    
    hessian.setAll(0.0);
    
    for(size_t i = 0; i < features.rows(); ++i)
    {
        // Calculate the Jacobian matrix of the base function at this point with
        // respect to the model parameters
        mBaseFunction.calculateJacobianParameters(features[i], jacobian);

        // Calculate the square of the Jacobian matrix. 
        // TODO: Calculate J^T and work with that for better cache performance
        // c1 -> r1, c2 -> r2, r -> c. Reverse the indices
        for (size_t c1 = 0; c1 < N; ++c1)
        {
            for (size_t c2 = 0; c2 < N; ++c2)
            {
                double sum = 0.0;
                for (size_t r = 0; r < M; ++r)
                    sum += jacobian[r][c1] * jacobian[r][c2];

                jacobianSquare[c1][c2] = sum;
            }
        }
        
        // Calculate the error for this sample
        if (mBaseFunction.cachesLastEvaluation())
            mBaseFunction.getLastEvaluation(evaluation);
        else mBaseFunction.evaluate(features[i], evaluation);
        
        for (size_t j = 0; j < M; ++j)
            error[j] = labels[i][j] - evaluation[j];
        
        // Calculate the sum of the local Hessians
        sumOfLocalHessians.setAll(0.0);
        for (size_t j = 0; j < M; ++j)
        {
            // Calculate the local Hessian for output j
            mBaseFunction.calculateHessianParameters(features[i], j, localHessian);
            
            // Multiply by the error constant and add to the running total
            for (size_t r = 0; r < N; ++r)
            {
                for (size_t c = 0; c < N; ++c)
                    sumOfLocalHessians[r][c] += error[j] * localHessian[r][c];
            }
        }
        
        // Finally, calculate the Hessian
        for (size_t r = 0; r < N; ++r)
        {
            for (size_t c = 0; c < N; ++c)
            {
                hessian[r][c] += 2.0 * 
                    (jacobianSquare[r][c] - sumOfLocalHessians[r][c]);
            }
        }
    }
}
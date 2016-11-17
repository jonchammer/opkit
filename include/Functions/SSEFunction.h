/* 
 * File:   SSEFunction.h
 * Author: Jon C. Hammer
 *
 * Created on August 9, 2016, 9:04 AM
 */

#ifndef SSEFUNCTION_H
#define SSEFUNCTION_H

#include <vector>
#include "ErrorFunction.h"
#include "Matrix.h"
#include "NeuralNetwork.h"
using std::vector;

namespace athena
{
    
// This class is an implementation of the SSE Error function.
template <class T>
class SSEFunction : public ErrorFunction<T>
{
public:
    SSEFunction(T& baseFunction) : ErrorFunction<T>(baseFunction)
    {
        // Do nothing
    }
    
    double evaluate(const Matrix& features, const Matrix& labels)
    {
        // Initialize variables
        double sum = 0.0;
        static vector<double> prediction(labels.cols(), 0.0);
        prediction.resize(labels.cols());

        // Calculate the SSE
        for (size_t i = 0; i < features.rows(); ++i)
        {
            ErrorFunction<T>::mBaseFunction.evaluate(features[i], prediction);

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
    
    void calculateGradientInputs(const Matrix& features, const Matrix& labels, 
        vector<double>& gradient)
    {
        // When SSE is the error function, the gradient is simply the error vector
        // multiplied by the model's Jacobian.
        const size_t N    = ErrorFunction<T>::mBaseFunction.getInputs();
        const size_t M    = ErrorFunction<T>::mBaseFunction.getOutputs();
        const size_t rows = features.rows();
        
        // Set the gradient to the zero vector
        gradient.resize(N);
        std::fill(gradient.begin(), gradient.end(), 0.0);

        static Matrix baseJacobian;
        static vector<double> evaluation(M);
        static vector<double> error(M);
        evaluation.resize(M);
        error.resize(M);

        for (size_t i = 0; i < rows; ++i)
        {
            // Calculate the Jacobian matrix of the base function at this point with
            // respect to the inputs
            ErrorFunction<T>::mBaseFunction.calculateJacobianInputs(features[i], baseJacobian);

            // Calculate the error for this sample
            if (ErrorFunction<T>::mBaseFunction.cachesLastEvaluation())
                ErrorFunction<T>::mBaseFunction.getLastEvaluation(evaluation);
            else ErrorFunction<T>::mBaseFunction.evaluate(features[i], evaluation);

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
        
        // Divide by the batch size to get the average gradient
        for (size_t i = 0; i < N; ++i)
            gradient[i] /= rows;
    }
    
    void calculateGradientParameters(const Matrix& features, 
        const Matrix& labels, vector<double>& gradient)
    {
        // When SSE is the error function, the gradient is simply the error vector
        // multiplied by the model's Jacobian.
        const size_t N    = ErrorFunction<T>::mBaseFunction.getNumParameters();
        const size_t M    = ErrorFunction<T>::mBaseFunction.getOutputs();
        const size_t rows = features.rows();
        
        // Set the gradient to the zero vector
        gradient.resize(N);
        std::fill(gradient.begin(), gradient.end(), 0.0);

        static Matrix baseJacobian;
        static vector<double> evaluation(M);
        static vector<double> error(M);
        evaluation.resize(M);
        error.resize(M);

        for (size_t i = 0; i < rows; ++i)
        {
            // Calculate the Jacobian matrix of the base function at this point with
            // respect to the model parameters
            ErrorFunction<T>::mBaseFunction.calculateJacobianParameters(features[i], baseJacobian);

            // Calculate the error for this sample
            if (ErrorFunction<T>::mBaseFunction.cachesLastEvaluation())
                ErrorFunction<T>::mBaseFunction.getLastEvaluation(evaluation);
            else ErrorFunction<T>::mBaseFunction.evaluate(features[i], evaluation);

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
        
        for (size_t i = 0; i < N; ++i)
            gradient[i] /= rows;
    }
    
    void calculateHessianInputs(const Matrix& features, const Matrix& labels,
        Matrix& hessian)
    {
        // When SSE is the error function, it is better to calculate the Hessian
        // directly using the following formula than to use finite differences.
        //
        // H = 2 * ((J^T * J) - sum_i((y_i - f(x_i, theta)) * H_i)), where
        //
        // H_i is the model's Hessian matrix with respect to output i
        // J is the model's Jacobian matrix

        const size_t N = ErrorFunction<T>::mBaseFunction.getInputs();
        const size_t M = ErrorFunction<T>::mBaseFunction.getOutputs();

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
            ErrorFunction<T>::mBaseFunction.calculateJacobianInputs(features[i], jacobian);

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
            if (ErrorFunction<T>::mBaseFunction.cachesLastEvaluation())
                ErrorFunction<T>::mBaseFunction.getLastEvaluation(evaluation);
            else ErrorFunction<T>::mBaseFunction.evaluate(features[i], evaluation);

            for (size_t j = 0; j < M; ++j)
                error[j] = labels[i][j] - evaluation[j];

            // Calculate the sum of the local Hessians
            sumOfLocalHessians.setAll(0.0);
            for (size_t j = 0; j < M; ++j)
            {
                // Calculate the local Hessian for output j
                ErrorFunction<T>::mBaseFunction.calculateHessianInputs(features[i], j, localHessian);

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
    
    void calculateHessianParameters(const Matrix& features, 
        const Matrix& labels, Matrix& hessian)
    {
        // When SSE is the error function, it is better to calculate the Hessian
        // directly using the following formula than to use finite differences.
        //
        // H = 2 * ((J^T * J) - sum_i((y_i - f(x_i, theta)) * H_i)), where
        //
        // H_i is the model's Hessian matrix with respect to output i
        // J is the model's Jacobian matrix

        const size_t N = ErrorFunction<T>::mBaseFunction.getNumParameters();
        const size_t M = ErrorFunction<T>::mBaseFunction.getOutputs();

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
            ErrorFunction<T>::mBaseFunction.calculateJacobianParameters(features[i], jacobian);

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
            if (ErrorFunction<T>::mBaseFunction.cachesLastEvaluation())
                ErrorFunction<T>::mBaseFunction.getLastEvaluation(evaluation);
            else ErrorFunction<T>::mBaseFunction.evaluate(features[i], evaluation);

            for (size_t j = 0; j < M; ++j)
                error[j] = labels[i][j] - evaluation[j];

            // Calculate the sum of the local Hessians
            sumOfLocalHessians.setAll(0.0);
            for (size_t j = 0; j < M; ++j)
            {
                // Calculate the local Hessian for output j
                ErrorFunction<T>::mBaseFunction.calculateHessianParameters(features[i], j, localHessian);

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
};

// Template specialization for Neural Networks, since there is a much more
// efficient mechanism for calculating the gradient with them.
template<>
class SSEFunction<NeuralNetwork> : public ErrorFunction<NeuralNetwork>
{
public:
    SSEFunction(NeuralNetwork& baseFunction) : ErrorFunction<NeuralNetwork>(baseFunction)
    {
        // Do nothing
    }
    
    double evaluate(const Matrix& features, const Matrix& labels)
    {
        // Initialize variables
        double sum = 0.0;
        static vector<double> prediction(labels.cols(), 0.0);
        prediction.resize(labels.cols());

        // Calculate the SSE
        for (size_t i = 0; i < features.rows(); ++i)
        {
            ErrorFunction<NeuralNetwork>::mBaseFunction.evaluate(features[i], prediction);

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
    
    void calculateGradientInputs(const Matrix& features, const Matrix& labels, 
        vector<double>& gradient)
    {
        NeuralNetwork& nn = ErrorFunction<NeuralNetwork>::mBaseFunction;
        const size_t N    = nn.getInputs();
        const size_t M    = nn.getOutputs();
        const size_t rows = features.rows();
        
        gradient.resize(N);
        std::fill(gradient.begin(), gradient.end(), 0.0);
        
        static vector<double> evaluation(M);
        static vector<double> tempGradient(N);
        evaluation.resize(M);
        tempGradient.resize(N);
        
        // Calculate a partial gradient for each row in the training data
        for (size_t i = 0; i < rows; ++i)
        {
            const vector<double>& feature = features[i];
            const vector<double>& label   = labels[i];
            
            // Forward prop
            nn.evaluate(feature, evaluation);
            
            // Calculate the deltas for each node in the network
            {
                // Calculate the deltas on the last layer first
                vector<double>& outputDeltas = nn.getOutputLayer()->getDeltas();
                for (size_t j = 0; j < outputDeltas.size(); ++j)
                    outputDeltas[j] = label[j] - evaluation[j];
                nn.getOutputLayer()->deactivateDeltas();

                // Apply the delta process recursively for each layer, moving backwards
                // through the network.
                for (int i = nn.getNumLayers() - 1; i >= 1; --i)
                {
                    Layer* current = nn.getLayer(i);
                    Layer* prev    = nn.getLayer(i - 1);
                    
                    current->calculateDeltas(prev->getDeltas());
                    prev->deactivateDeltas();
                }
            }
            
            // Calculate the gradient based on the deltas. Values are summed
            // for each pattern.
            nn.getLayer(0)->calculateDeltas(tempGradient);
            
            // Technically, we need to multiply the final gradient by a factor
            // of -2 to get the true gradient with respect to the SSE function.
            for (size_t j = 0; j < tempGradient.size(); ++j)
                gradient[j] += -2.0 * tempGradient[j];
        } 
        
        // Divide by the batch size to get the average gradient
        for (size_t i = 0; i < N; ++i)
            gradient[i] /= rows;
    }
    
    void calculateGradientParameters(const Matrix& features, 
        const Matrix& labels, vector<double>& gradient)
    {
        NeuralNetwork& nn = ErrorFunction<NeuralNetwork>::mBaseFunction;
        const size_t N    = nn.getNumParameters();
        const size_t M    = nn.getOutputs();
        const size_t rows = features.rows();
        
        gradient.resize(N);
        std::fill(gradient.begin(), gradient.end(), 0.0);
        
        static vector<double> evaluation(M);
        evaluation.resize(M);
        
        // Calculate a partial gradient for each row in the training data
        for (size_t i = 0; i < rows; ++i)
        {
            const vector<double>& feature = features[i];
            const vector<double>& label   = labels[i];
            
            // Forward prop
            nn.evaluate(feature, evaluation);
            
            // Calculate the deltas for each node in the network
            {
                // Calculate the deltas on the last layer first
                vector<double>& outputDeltas = nn.getOutputLayer()->getDeltas();
                for (size_t j = 0; j < outputDeltas.size(); ++j)
                    outputDeltas[j] = label[j] - evaluation[j];
                nn.getOutputLayer()->deactivateDeltas();

                // Apply the delta process recursively for each layer, moving backwards
                // through the network.
                for (int i = nn.getNumLayers() - 1; i >= 1; --i)
                {
                    Layer* current = nn.getLayer(i);
                    Layer* prev    = nn.getLayer(i - 1);
                    
                    current->calculateDeltas(prev->getDeltas());
                    prev->deactivateDeltas();
                }
            }
            
            // Calculate the gradient based on the deltas. Values are summed
            // for each pattern.
            nn.calculateGradientParameters(feature, gradient);
        }
        
        // Technically, we need to multiply the final gradient by a factor
        // of -2 to get the true gradient with respect to the SSE function.
        // We also need to divide by the batch size to get an average gradient.
        for (size_t j = 0; j < gradient.size(); ++j)
        {
            gradient[j] *= (-2.0 / rows);
        }
    }
    
    void calculateHessianInputs(const Matrix& features, const Matrix& labels,
        Matrix& hessian)
    {
        // When SSE is the error function, it is better to calculate the Hessian
        // directly using the following formula than to use finite differences.
        //
        // H = 2 * ((J^T * J) - sum_i((y_i - f(x_i, theta)) * H_i)), where
        //
        // H_i is the model's Hessian matrix with respect to output i
        // J is the model's Jacobian matrix

        const size_t N = ErrorFunction<NeuralNetwork>::mBaseFunction.getInputs();
        const size_t M = ErrorFunction<NeuralNetwork>::mBaseFunction.getOutputs();

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
            ErrorFunction<NeuralNetwork>::mBaseFunction.calculateJacobianInputs(features[i], jacobian);

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
            if (ErrorFunction<NeuralNetwork>::mBaseFunction.cachesLastEvaluation())
                ErrorFunction<NeuralNetwork>::mBaseFunction.getLastEvaluation(evaluation);
            else ErrorFunction<NeuralNetwork>::mBaseFunction.evaluate(features[i], evaluation);

            for (size_t j = 0; j < M; ++j)
                error[j] = labels[i][j] - evaluation[j];

            // Calculate the sum of the local Hessians
            sumOfLocalHessians.setAll(0.0);
            for (size_t j = 0; j < M; ++j)
            {
                // Calculate the local Hessian for output j
                ErrorFunction<NeuralNetwork>::mBaseFunction.calculateHessianInputs(features[i], j, localHessian);

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
    
    void calculateHessianParameters(const Matrix& features, 
        const Matrix& labels, Matrix& hessian)
    {
        // When SSE is the error function, it is better to calculate the Hessian
        // directly using the following formula than to use finite differences.
        //
        // H = 2 * ((J^T * J) - sum_i((y_i - f(x_i, theta)) * H_i)), where
        //
        // H_i is the model's Hessian matrix with respect to output i
        // J is the model's Jacobian matrix

        const size_t N = ErrorFunction<NeuralNetwork>::mBaseFunction.getNumParameters();
        const size_t M = ErrorFunction<NeuralNetwork>::mBaseFunction.getOutputs();

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
            ErrorFunction<NeuralNetwork>::mBaseFunction.calculateJacobianParameters(features[i], jacobian);

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
            if (ErrorFunction<NeuralNetwork>::mBaseFunction.cachesLastEvaluation())
                ErrorFunction<NeuralNetwork>::mBaseFunction.getLastEvaluation(evaluation);
            else ErrorFunction<NeuralNetwork>::mBaseFunction.evaluate(features[i], evaluation);

            for (size_t j = 0; j < M; ++j)
                error[j] = labels[i][j] - evaluation[j];

            // Calculate the sum of the local Hessians
            sumOfLocalHessians.setAll(0.0);
            for (size_t j = 0; j < M; ++j)
            {
                // Calculate the local Hessian for output j
                ErrorFunction<NeuralNetwork>::mBaseFunction.calculateHessianParameters(features[i], j, localHessian);

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
};

};

#endif /* SSEFUNCTION_H */


/*
 * File:   SSEFunction.h
 * Author: Jon C. Hammer
 *
 * Created on August 9, 2016, 9:04 AM
 */

#ifndef SSEFUNCTION_H
#define SSEFUNCTION_H

#include <vector>
#include <cstring>
#include "ErrorFunction.h"
#include "Dataset.h"
#include "Matrix.h"
#include "NeuralNetwork.h"
#include "Acceleration.h"
#include "PrettyPrinter.h"
using std::vector;

namespace opkit
{

// This class is an implementation of the SSE Error function.
template <class T, class Model>
class SSEFunction : public ErrorFunction<T, Model>
{
public:

    using ErrorFunction<T, Model>::mBaseFunction;

    SSEFunction(Model& baseFunction) : ErrorFunction<T, Model>(baseFunction)
    {
        // Do nothing
    }

    T evaluate(const Dataset<T>& features, const Dataset<T>& labels)
    {
        // Initialize variables
        T sum{};
        static vector<T> prediction(labels.cols(), T{});

        // Calculate the SSE
        for (size_t i = 0; i < features.rows(); ++i)
        {
            mBaseFunction.evaluate(features[i], prediction);

            const vector<T>& row = labels[i];
            for (size_t j = 0; j < labels.cols(); ++j)
            {
                T d = row[j] - prediction[j];

                // For categorical columns, use Hamming distance instead
                //if (d != 0.0 && labels.valueCount(j) > 0)
                //    d = 1.0;

                sum += (d * d);
            }
        }

        return sum;
    }

    void calculateGradientInputs(const Dataset<T>& features, const Dataset<T>& labels,
        vector<T>& gradient)
    {
        // When SSE is the error function, the gradient is simply the error vector
        // multiplied by the model's Jacobian.
        const size_t N    = mBaseFunction.getInputs();
        const size_t M    = mBaseFunction.getOutputs();
        const size_t rows = features.rows();

        // Set the gradient to the zero vector
        std::fill(gradient.begin(), gradient.end(), T{});

        static Matrix<T> baseJacobian(M, N);
        static vector<T> evaluation(M);
        static Matrix<T> error(1, M);

        // The matrix 'grad' temporarily holds the contents of the gradient
        static Matrix<T> grad(1, N);
        grad.swap(gradient);

        for (size_t i = 0; i < rows; ++i)
        {
            // Calculate the Jacobian matrix of the base function at this point with
            // respect to the inputs
            mBaseFunction.calculateJacobianInputs(features[i], baseJacobian);

            // Calculate the error for this sample
            if (mBaseFunction.cachesLastEvaluation())
                mBaseFunction.getLastEvaluation(evaluation);
            else mBaseFunction.evaluate(features[i], evaluation);

            for (size_t j = 0; j < M; ++j)
                error(0, j) = labels[i][j] - evaluation[j];

            grad += T(-2.0) * error * baseJacobian;
        }

        // Swap back so gradient contains the correct information
        grad.swap(gradient);

        // Divide by the batch size to get the average gradient
        for (size_t i = 0; i < N; ++i)
            gradient[i] /= rows;
    }

    void calculateGradientParameters(const Dataset<T>& features,
        const Dataset<T>& labels, vector<T>& gradient)
    {
        // When SSE is the error function, the gradient is simply the error vector
        // multiplied by the model's Jacobian.
        const size_t N    = mBaseFunction.getNumParameters();
        const size_t M    = mBaseFunction.getOutputs();
        const size_t rows = features.rows();

        // Set the gradient to the zero vector
        std::fill(gradient.begin(), gradient.end(), T{});

        static Matrix<T> baseJacobian(M, N);
        static vector<T> evaluation(M);
        static Matrix<T> error(1, M);

        // The matrix 'grad' temporarily holds the contents of the gradient
        static Matrix<T> grad(1, N);
        grad.swap(gradient);

        for (size_t i = 0; i < rows; ++i)
        {
            // Calculate the Jacobian matrix of the base function at this point with
            // respect to the model parameters
            mBaseFunction.calculateJacobianParameters(features[i], baseJacobian);

            // Calculate the error for this sample
            if (mBaseFunction.cachesLastEvaluation())
                mBaseFunction.getLastEvaluation(evaluation);
            else mBaseFunction.evaluate(features[i], evaluation);

            for (size_t j = 0; j < M; ++j)
                error(0, j) = labels[i][j] - evaluation[j];

            grad += T(-2.0) * error * baseJacobian;
        }

        // Swap back so gradient contains the correct information
        grad.swap(gradient);

        for (size_t i = 0; i < N; ++i)
            gradient[i] /= rows;
    }

    void calculateHessianInputs(const Dataset<T>& features,
        const Dataset<T>& labels, Matrix<T>& hessian)
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
        Matrix<T> jacobian(M, N);
        Matrix<T> localHessian(N, N);
        Matrix<T> sumOfLocalHessians(N, N);
        Matrix<T> error(1, M);
        vector<T> evaluation(M);

        hessian.resize(N, N);
        hessian.fill(T{});

        for(size_t i = 0; i < features.rows(); ++i)
        {
            // Calculate the Jacobian matrix of the base function at this point with
            // respect to the model parameters
            mBaseFunction.calculateJacobianInputs(features[i], jacobian);

            // Calculate the error for this sample
            if (mBaseFunction.cachesLastEvaluation())
                mBaseFunction.getLastEvaluation(evaluation);
            else mBaseFunction.evaluate(features[i], evaluation);

            for (size_t j = 0; j < M; ++j)
                error(0, j) = labels[i][j] - evaluation[j];

            // Calculate the sum of the local Hessians
            sumOfLocalHessians.fill(T{});
            for (size_t j = 0; j < M; ++j)
            {
                // Calculate the local Hessian for output j and multiply by the
                // error for this output
                mBaseFunction.calculateHessianInputs(features[i], j, localHessian);
                sumOfLocalHessians += error(0, j) * localHessian;
            }

            // Finally, calculate the Hessian
            hessian += T(2.0) * (transpose(jacobian) * jacobian - sumOfLocalHessians);
        }
    }

    void calculateHessianParameters(const Dataset<T>& features,
        const Dataset<T>& labels, Matrix<T>& hessian)
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
        Matrix<T> jacobian(M, N);
        Matrix<T> localHessian(N, N);
        Matrix<T> sumOfLocalHessians(N, N);
        Matrix<T> error(1, M);
        vector<T> evaluation(M);

        hessian.resize(N, N);
        hessian.fill(T{});

        for(size_t i = 0; i < features.rows(); ++i)
        {
            // Calculate the Jacobian matrix of the base function at this point with
            // respect to the model parameters
            mBaseFunction.calculateJacobianParameters(features[i], jacobian);

            // Calculate the error for this sample
            if (mBaseFunction.cachesLastEvaluation())
                mBaseFunction.getLastEvaluation(evaluation);
            else mBaseFunction.evaluate(features[i], evaluation);

            for (size_t j = 0; j < M; ++j)
                error(0, j) = labels[i][j] - evaluation[j];

            // Calculate the sum of the local Hessians
            sumOfLocalHessians.fill(T{});
            for (size_t j = 0; j < M; ++j)
            {
                // Calculate the local Hessian for output j and multiply by the
                // error for this output
                mBaseFunction.calculateHessianParameters(features[i], j, localHessian);
                sumOfLocalHessians += error(0, j) * localHessian;
            }

            // Finally, calculate the Hessian
            hessian += T(2.0) * (transpose(jacobian) * jacobian - sumOfLocalHessians);
        }
    }
};

// Template specialization for Neural Networks, since there is a much more
// efficient mechanism for calculating the gradient with them.
template<class T>
class SSEFunction<T, NeuralNetwork<T>> : public ErrorFunction<T, NeuralNetwork<T>>
{
public:

    using ErrorFunction<T, NeuralNetwork<T>>::mBaseFunction;

    SSEFunction(NeuralNetwork<T>& baseFunction) : ErrorFunction<T, NeuralNetwork<T>>(baseFunction)
    {
        // Do nothing
    }

    T evaluate(const Dataset<T>& features, const Dataset<T>& labels)
    {
        // Initialize variables
        const size_t N = features.rows();
        const size_t M = labels.cols();

        T sum {};
        static vector<T> prediction(M, T{});

        // Calculate the SSE
        for (size_t i = 0; i < N; ++i)
        {
            mBaseFunction.evaluate(features[i], prediction);

            const vector<T>& row = labels[i];
            for (size_t j = 0; j < M; ++j)
            {
                T d = row[j] - prediction[j];

                // For categorical columns, use Hamming distance instead
                //if (d != 0.0 && labels.valueCount(j) > 0)
                //    d = 1.0;

                sum += (d * d);
            }
        }

        return sum;
    }

    void calculateGradientInputs(const Dataset<T>& features,
        const Dataset<T>& labels, vector<T>& gradient)
    {
        NeuralNetwork<T>& nn = mBaseFunction;
        const size_t N       = nn.getInputs();
        const size_t M       = nn.getOutputs();
        const size_t rows    = features.rows();

        std::fill(gradient.begin(), gradient.end(), T{});

        static vector<T> evaluation(M);
        static vector<T> tempGradient(N);

        // Calculate a partial gradient for each row in the training data
        for (size_t i = 0; i < rows; ++i)
        {
            const vector<T>& feature = features[i];
            const vector<T>& label   = labels[i];

            // Forward prop
            nn.evaluate(feature, evaluation);

            // Calculate the deltas for each node in the network
            {
                // Calculate the deltas on the last layer first
                vector<T>& outputDeltas = nn.getOutputLayer()->getDeltas();
                for (size_t j = 0; j < outputDeltas.size(); ++j)
                    outputDeltas[j] = label[j] - evaluation[j];

                nn.calculateDeltas();
            }

            // Calculate the gradient based on the deltas. Values are summed
            // for each pattern.
            nn.getLayer(0)->calculateDeltas(feature, tempGradient);

            // Technically, we need to multiply the final gradient by a factor
            // of -2 to get the true gradient with respect to the SSE function.
            vAdd(tempGradient.data(), gradient.data(), N, -2.0);
        }

        // Divide by the batch size to get the average gradient
        vScale(gradient.data(), 1.0/rows, N);
    }

    void calculateGradientParameters(const Dataset<T>& features,
        const Dataset<T>& labels, vector<T>& gradient)
    {
        NeuralNetwork<T>& nn = mBaseFunction;
        const size_t N       = nn.getNumParameters();
        const size_t M       = nn.getOutputs();
        const size_t rows    = features.rows();

        std::fill(gradient.begin(), gradient.end(), T{});
        static vector<T> evaluation(M);

        // Calculate a partial gradient for each row in the training data
        for (size_t i = 0; i < rows; ++i)
        {
            const vector<T>& feature = features[i];
            const vector<T>& label   = labels[i];

            // Forward prop
            nn.evaluate(feature, evaluation);

            // Calculate the deltas for each node in the network
            {
                // Calculate the deltas on the last layer first
                vector<T>& outputDeltas = nn.getOutputLayer()->getDeltas();
                for (size_t j = 0; j < M; ++j)
                    outputDeltas[j] = label[j] - evaluation[j];

                nn.calculateDeltas();
            }

            // Calculate the gradient based on the deltas. Values are summed
            // for each pattern.
            nn.calculateGradientParameters(feature, gradient);
        }

        // Technically, we need to multiply the final gradient by a factor
        // of -2 to get the true gradient with respect to the SSE function.
        // We also need to divide by the batch size to get an average gradient.
        vScale(gradient.data(), -2.0/rows, N);
    }

    void calculateHessianInputs(const Dataset<T>& features,
        const Dataset<T>& labels, Matrix<T>& hessian)
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
        Matrix<T> jacobian(M, N);
        Matrix<T> localHessian(N, N);
        Matrix<T> sumOfLocalHessians(N, N);
        Matrix<T> error(1, M);
        vector<T> evaluation(M);

        hessian.resize(N, N);
        hessian.fill(T{});

        for(size_t i = 0; i < features.rows(); ++i)
        {
            // Calculate the Jacobian matrix of the base function at this point with
            // respect to the model parameters
            mBaseFunction.calculateJacobianInputs(features[i], jacobian);

            // Calculate the error for this sample
            if (mBaseFunction.cachesLastEvaluation())
                mBaseFunction.getLastEvaluation(evaluation);
            else mBaseFunction.evaluate(features[i], evaluation);

            for (size_t j = 0; j < M; ++j)
                error(0, j) = labels[i][j] - evaluation[j];

            // Calculate the sum of the local Hessians
            sumOfLocalHessians.fill(T{});
            for (size_t j = 0; j < M; ++j)
            {
                // Calculate the local Hessian for output j
                mBaseFunction.calculateHessianInputs(features[i], j, localHessian);
                sumOfLocalHessians += error(0, j) * localHessian;
            }

            // Finally, calculate the Hessian
            hessian += 2.0 * (transpose(jacobian) * jacobian - sumOfLocalHessians);
        }
    }

    void calculateHessianParameters(const Dataset<T>& features,
        const Dataset<T>& labels, Matrix<T>& hessian)
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
        Matrix<T> jacobian(M, N);
        Matrix<T> jacobianSquare(N, N);
        Matrix<T> localHessian(N, N);
        Matrix<T> sumOfLocalHessians(N, N);
        Matrix<T> error(1, M);
        vector<T> evaluation(M);

        hessian.resize(N, N);
        hessian.fill(T{});

        for(size_t i = 0; i < features.rows(); ++i)
        {
            // Calculate the Jacobian matrix of the base function at this point with
            // respect to the model parameters
            mBaseFunction.calculateJacobianParameters(features[i], jacobian);

            // Calculate the error for this sample
            if (mBaseFunction.cachesLastEvaluation())
                mBaseFunction.getLastEvaluation(evaluation);
            else mBaseFunction.evaluate(features[i], evaluation);

            for (size_t j = 0; j < M; ++j)
                error(0, j) = labels[i][j] - evaluation[j];

            // Calculate the sum of the local Hessians
            sumOfLocalHessians.fill(T{});
            for (size_t j = 0; j < M; ++j)
            {
                // Calculate the local Hessian for output j
                mBaseFunction.calculateHessianParameters(features[i], j, localHessian);
                sumOfLocalHessians += error(0, j) * localHessian;
            }

            // Finally, calculate the Hessian
            hessian += 2.0 * (transpose(jacobian) * jacobian - sumOfLocalHessians);
        }
    }
};

};

#endif /* SSEFUNCTION_H */

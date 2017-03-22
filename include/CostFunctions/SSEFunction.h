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
#include <cassert>
#include "CostFunction.h"
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
class SSEFunction : public CostFunction<T, Model>
{
public:

    using CostFunction<T, Model>::mBaseFunction;

    SSEFunction(Model& baseFunction) : CostFunction<T, Model>(baseFunction)
    {
        // Do nothing
    }

    T evaluate(const Matrix<T>& features, const Matrix<T>& labels)
    {
        // Initialize variables
        T sum{};
        static vector<T> prediction(labels.getCols(), T{});

        // Calculate the SSE
        for (size_t i = 0; i < features.getRows(); ++i)
        {
            mBaseFunction.evaluate(features(i), prediction.data());

            for (size_t j = 0; j < labels.getCols(); ++j)
            {
                T d = labels(i, j) - prediction[j];
                sum += (d * d);
            }
        }

        return sum;
    }

    void calculateGradientInputs(const Matrix<T>& features, const Matrix<T>& labels,
        vector<T>& gradient)
    {
        // When SSE is the error function, the gradient is simply the error vector
        // multiplied by the model's Jacobian.
        const size_t N    = mBaseFunction.getInputs();
        const size_t M    = mBaseFunction.getOutputs();
        const size_t rows = features.getRows();

        // Make sure gradient has enough space
        assert(gradient.size() >= N);

        // Set the gradient to the zero vector
        std::fill(gradient.begin(), gradient.end(), T{});

        static Matrix<T> baseJacobian(M, N);
        static vector<T> evaluation(M);
        static Matrix<T> error(1, M);

        // The matrix 'grad' temporarily holds the contents of the gradient
        Matrix<T> grad(gradient.data(), 1, N);

        for (size_t i = 0; i < rows; ++i)
        {
            // Calculate the Jacobian matrix of the base function at this point with
            // respect to the inputs
            mBaseFunction.calculateJacobianInputs(features(i), baseJacobian);

            // Calculate the error for this sample
            mBaseFunction.evaluate(features(i), evaluation.data());

            for (size_t j = 0; j < M; ++j)
                error(0, j) = labels(i, j) - evaluation[j];

            grad += T{-2.0} * error * baseJacobian;
        }

        // Divide by the batch size to get the average gradient
        vScale(gradient.data(), 1.0/rows, N);
    }

    void calculateGradientParameters(const Matrix<T>& features,
        const Matrix<T>& labels, vector<T>& gradient)
    {
        // When SSE is the error function, the gradient is simply the error vector
        // multiplied by the model's Jacobian.
        const size_t N    = mBaseFunction.getNumParameters();
        const size_t M    = mBaseFunction.getOutputs();
        const size_t rows = features.getRows();

        // Make sure gradient has enough space
        assert(gradient.size() >= N);

        // Set the gradient to the zero vector
        std::fill(gradient.begin(), gradient.end(), T{});

        static Matrix<T> baseJacobian(M, N);
        static vector<T> evaluation(M);
        static Matrix<T> error(1, M);

        // The matrix 'grad' temporarily holds the contents of the gradient
        Matrix<T> grad(gradient.data(), 1, N);

        for (size_t i = 0; i < rows; ++i)
        {
            // Calculate the Jacobian matrix of the base function at this point with
            // respect to the model parameters
            mBaseFunction.calculateJacobianParameters(features(i), baseJacobian);

            // Calculate the error for this sample
            mBaseFunction.evaluate(features(i), evaluation.data());

            for (size_t j = 0; j < M; ++j)
                error(0, j) = labels(i, j) - evaluation[j];

            grad += T{-2.0} * error * baseJacobian;
        }

        vScale(gradient.data(), 1.0/rows, N);
    }

    void calculateHessianInputs(const Matrix<T>& features,
        const Matrix<T>& labels, Matrix<T>& hessian)
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

        for(size_t i = 0; i < features.getRows(); ++i)
        {
            // Calculate the Jacobian matrix of the base function at this point with
            // respect to the model parameters
            mBaseFunction.calculateJacobianInputs(features(i), jacobian);

            // Calculate the error for this sample
            mBaseFunction.evaluate(features(i), evaluation.data());

            for (size_t j = 0; j < M; ++j)
                error(0, j) = labels(i, j) - evaluation[j];

            // Calculate the sum of the local Hessians
            sumOfLocalHessians.fill(T{});
            for (size_t j = 0; j < M; ++j)
            {
                // Calculate the local Hessian for output j and multiply by the
                // error for this output
                mBaseFunction.calculateHessianInputs(features(i), j, localHessian);
                sumOfLocalHessians += error(0, j) * localHessian;
            }

            // Finally, calculate the Hessian
            hessian += T{2.0} * (transpose(jacobian) * jacobian - sumOfLocalHessians);
        }
    }

    void calculateHessianParameters(const Matrix<T>& features,
        const Matrix<T>& labels, Matrix<T>& hessian)
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

        for(size_t i = 0; i < features.getRows(); ++i)
        {
            // Calculate the Jacobian matrix of the base function at this point with
            // respect to the model parameters
            mBaseFunction.calculateJacobianParameters(features(i), jacobian);

            // Calculate the error for this sample
            mBaseFunction.evaluate(features(i), evaluation.data());

            for (size_t j = 0; j < M; ++j)
                error(0, j) = labels(i, j) - evaluation[j];

            // Calculate the sum of the local Hessians
            sumOfLocalHessians.fill(T{});
            for (size_t j = 0; j < M; ++j)
            {
                // Calculate the local Hessian for output j and multiply by the
                // error for this output
                mBaseFunction.calculateHessianParameters(features(i), j, localHessian);
                sumOfLocalHessians += error(0, j) * localHessian;
            }

            // Finally, calculate the Hessian
            hessian += T{2.0} * (transpose(jacobian) * jacobian - sumOfLocalHessians);
        }
    }
};

// Template specialization for Neural Networks, since there is a much more
// efficient mechanism for calculating the gradient with them.
template<class T>
class SSEFunction<T, NeuralNetwork<T>> : public CostFunction<T, NeuralNetwork<T>>
{
public:

    using CostFunction<T, NeuralNetwork<T>>::mBaseFunction;

    SSEFunction(NeuralNetwork<T>& baseFunction) : CostFunction<T, NeuralNetwork<T>>(baseFunction)
    {
        // Do nothing
    }

    T evaluate(const Matrix<T>& features, const Matrix<T>& labels)
    {
        // Initialize variables
        const size_t batchSize = mBaseFunction.getMaxBatchSize();
        const size_t M         = features.getCols();
        const size_t N         = labels.getCols();

        Matrix<T> batchFeatures((T*) features.data(), batchSize, M);
        Matrix<T> batchLabels((T*) labels.data(), batchSize, N);
        static Matrix<T> predictions(batchSize, N);

        T sum{};

        // Calculate the SSE
        size_t rows = features.getRows();
        while (rows >= batchSize)
        {
            sum += evalBatch(batchFeatures, batchLabels, predictions);

            // Move to the next batch
            T* featureData = batchFeatures.data();
            T* labelData   = batchLabels.data();
            batchFeatures.setData(featureData + batchSize * M);
            batchLabels.setData(labelData + batchSize * N);

            rows -= batchSize;
        }

        // Deal with the leftover elements
        if (rows > 0)
        {
            batchFeatures.reshape(rows, M);
            batchLabels.reshape(rows, N);
            predictions.reshape(rows, N);

            sum += evalBatch(batchFeatures, batchLabels, predictions);
        }

        return sum;
    }

private:

    // Measures the SSE for a single batch
    T evalBatch(Matrix<T>& batchFeatures, Matrix<T>& batchLabels,
        Matrix<T>& predictions)
    {
        const size_t batchSize = batchFeatures.getRows();
        const size_t N         = batchLabels.getCols();

        // Evaluate this minibatch
        mBaseFunction.evaluateBatch(batchFeatures, predictions);

        T sum {};
        for (size_t i = 0; i < batchSize; ++i)
        {
            for (size_t j = 0; j < N; ++j)
            {
                T d = batchLabels(i, j) - predictions(i, j);
                sum += (d * d);
            }
        }

        return sum;
    }

public:
    void calculateGradientInputs(const Matrix<T>& features,
        const Matrix<T>& labels, vector<T>& gradient)
    {
        NeuralNetwork<T>& nn = mBaseFunction;
        const size_t N       = nn.getInputs();
        const size_t M       = nn.getOutputs();
        const size_t rows    = features.getRows();

        // Make sure gradient has enough space
        assert(gradient.size() >= N);

        // Forward prop the training data through the network
        static Matrix<T> evaluation(rows, M);
        nn.evaluateBatch(features, evaluation);

        // Calculate the deltas for the last layer by subtracting the evaluation
        // from the labels.
        Matrix<T>& outputDeltas = nn.getOutputDeltas();
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < M; ++j)
                outputDeltas(i, j) = labels(i, j) - evaluation(i, j);
        }

        // Propagate the deltas back through the network
        nn.backpropInputsBatch();

        // To get the gradient with respect to the inputs, we need to propagate
        // the deltas in the first layer. This will give us a matrix of all the
        // gradient values. We will need to flatten this to a vector by
        // averaging the values across columns.
        Layer<T>* front = nn.getLayer(0);

        static Matrix<T> localGradients(rows, front->getOutputs());
        front->backpropInputsBatch(features, localGradients.data());

        // Average gradients across the columns
        for (size_t i = 0; i < rows; ++i)
            vAdd(localGradients(i), gradient.data(), N);
        vScale(gradient.data(), T{1.0} / rows, N);
    }

    void calculateGradientParameters(const Matrix<T>& features,
        const Matrix<T>& labels, vector<T>& gradient)
    {
        NeuralNetwork<T>& nn = mBaseFunction;
        const size_t N       = nn.getNumParameters();
        const size_t M       = nn.getOutputs();
        const size_t rows    = features.getRows();

        // Make sure gradient has enough space
        assert(gradient.size() >= N);

        // Forward prop the training data through the network
        static Matrix<T> evaluation(rows, M);
        nn.evaluateBatch(features, evaluation);

        // Calculate the deltas for the last layer by subtracting the evaluation
        // from the labels.
        Matrix<T>& outputDeltas = nn.getOutputDeltas();
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < M; ++j)
                outputDeltas(i, j) = labels(i, j) - evaluation(i, j);
        }

        // Propagate the deltas back through the network.
        nn.backpropInputsBatch();

        // Calculate the local gradients for each sample.
        static Matrix<T> localGradients(rows, N);
        nn.backpropParametersBatch(features, localGradients);

        // Average gradients across the columns.
        // Technically, we need to multiply the final gradient by a factor
        // of -2 to get the true gradient with respect to the SSE function.
        for (size_t i = 0; i < rows; ++i)
            vAdd(localGradients(i), gradient.data(), N);
        vScale(gradient.data(), T{-2.0} / rows, N);
    }

    void calculateHessianInputs(const Matrix<T>& features,
        const Matrix<T>& labels, Matrix<T>& hessian)
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

        for(size_t i = 0; i < features.getRows(); ++i)
        {
            // Calculate the Jacobian matrix of the base function at this point with
            // respect to the model parameters
            mBaseFunction.calculateJacobianInputs(features(i), jacobian);

            // Calculate the error for this sample
            mBaseFunction.evaluate(features(i), evaluation.data());

            for (size_t j = 0; j < M; ++j)
                error(0, j) = labels(i, j) - evaluation[j];

            // Calculate the sum of the local Hessians
            sumOfLocalHessians.fill(T{});
            for (size_t j = 0; j < M; ++j)
            {
                // Calculate the local Hessian for output j
                mBaseFunction.calculateHessianInputs(features(i), j, localHessian);
                sumOfLocalHessians += error(0, j) * localHessian;
            }

            // Finally, calculate the Hessian
            hessian += T{2.0} * (transpose(jacobian) * jacobian - sumOfLocalHessians);
        }
    }

    void calculateHessianParameters(const Matrix<T>& features,
        const Matrix<T>& labels, Matrix<T>& hessian)
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

        for(size_t i = 0; i < features.getRows(); ++i)
        {
            // Calculate the Jacobian matrix of the base function at this point with
            // respect to the model parameters
            mBaseFunction.calculateJacobianParameters(features(i), jacobian);

            // Calculate the error for this sample
            mBaseFunction.evaluate(features(i), evaluation.data());

            for (size_t j = 0; j < M; ++j)
                error(0, j) = labels(i, j) - evaluation[j];

            // Calculate the sum of the local Hessians
            sumOfLocalHessians.fill(T{});
            for (size_t j = 0; j < M; ++j)
            {
                // Calculate the local Hessian for output j
                mBaseFunction.calculateHessianParameters(features(i), j, localHessian);
                sumOfLocalHessians += error(0, j) * localHessian;
            }

            // Finally, calculate the Hessian
            hessian += T{2.0} * (transpose(jacobian) * jacobian - sumOfLocalHessians);
        }
    }
};

};

#endif /* SSEFUNCTION_H */

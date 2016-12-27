/*
 * File:   CrossEntropyFunction.h
 * Author: Jon C. Hammer
 *
 * Created on December 26, 2016, 10:59 AM
 */

#ifndef CROSSENTROPYFUNCTION_H
#define CROSSENTROPYFUNCTION_H

#include <vector>
#include <cmath>
#include "ErrorFunction.h"
#include "Matrix.h"
#include "NeuralNetwork.h"
#include "Acceleration.h"
#include "PrettyPrinter.h"
using std::vector;

namespace opkit
{

// This class is an implementation of the Cross-entropy Error function.
// NOTE: When using this class, the user must guarantee that the model produces
// only positive numbers. Negative numbers will cause the function to produce
// bad results (e.g. NaNs or +- infinity).
template <class T, class Model>
class CrossEntropyFunction : public ErrorFunction<T, Model>
{
public:

    using ErrorFunction<T, Model>::mBaseFunction;

    CrossEntropyFunction(Model& baseFunction) : ErrorFunction<T, Model>(baseFunction)
    {
        // Do nothing
    }

    T evaluate(const Matrix<T>& features, const Matrix<T>& labels)
    {
        // A very small number is added to the input to prevent log(0) becoming NaN.
        const T EPSILON = std::numeric_limits<T>::epsilon();
        const size_t N  = features.rows();
        const size_t M  = labels.cols();
        static vector<T> prediction(M);

        T sum{};
        for (size_t i = 0; i < N; ++i)
        {
            mBaseFunction.evaluate(features[i], prediction);

            const vector<T>& row = labels[i];
            for (size_t j = 0; j < M; ++j)
                sum += row[j] * std::log(prediction[j] + EPSILON);
        }

        return -sum;
    }

    void calculateGradientInputs(const Matrix<T>& features, const Matrix<T>& labels,
        vector<T>& gradient)
    {
        // When SSE is the error function, the gradient is simply the error vector
        // multiplied by the model's Jacobian.
        const size_t N    = mBaseFunction.getInputs();
        const size_t M    = mBaseFunction.getOutputs();
        const size_t rows = features.rows();

        // Set the gradient to the zero vector
        std::fill(gradient.begin(), gradient.end(), T{});

        static Matrix<T> baseJacobian;
        static vector<T> evaluation(M);
        static vector<T> error(M);

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
                error[j] = -labels[i][j] / evaluation[j];

            for (size_t j = 0; j < N; ++j)
            {
                // Multiply the error by the model's Jacobian,
                T sum{};
                for (size_t k = 0; k < M; ++k)
                    sum += error[k] * baseJacobian[k][j];

                // Add the result to the running total for the gradient
                gradient[j] += sum;
            }
        }

        // Divide by the batch size to get the average gradient
        vScale(gradient.data(), 1.0/rows, N);
    }

    void calculateGradientParameters(const Matrix<T>& features,
        const Matrix<T>& labels, vector<T>& gradient)
    {
        // When Cross-entropy is the error function, the gradient is equal to
        // -y'/y * the model's jacobian, where y' is the training sample and
        // y is the model's output.
        const size_t N    = mBaseFunction.getNumParameters();
        const size_t M    = mBaseFunction.getOutputs();
        const size_t rows = features.rows();

        // Set the gradient to the zero vector
        std::fill(gradient.begin(), gradient.end(), T{});

        static Matrix<T> baseJacobian;
        static vector<T> evaluation(M);
        static vector<T> error(M);

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
                error[j] = -labels[i][j] / evaluation[j];

            for (size_t j = 0; j < N; ++j)
            {
                // Multiply the error by the model's Jacobian,
                T sum{};
                for (size_t k = 0; k < M; ++k)
                    sum += error[k] * baseJacobian[k][j];

                // Add the result to the running total for the gradient
                gradient[j] +=  sum;
            }
        }

        vScale(gradient.data(), 1.0/rows, N);
    }

    void calculateHessianInputs(const Matrix<T>& features, const Matrix<T>& labels,
        Matrix<T>& hessian)
    {
        // TODO
    }

    void calculateHessianParameters(const Matrix<T>& features,
        const Matrix<T>& labels, Matrix<T>& hessian)
    {
        // TODO
    }
};

// Template specialization for Neural Networks, since there is a much more
// efficient mechanism for calculating the gradient with them.
template<class T>
class CrossEntropyFunction<T, NeuralNetwork<T>> : public ErrorFunction<T, NeuralNetwork<T>>
{
public:

    using ErrorFunction<T, NeuralNetwork<T>>::mBaseFunction;

    CrossEntropyFunction(NeuralNetwork<T>& baseFunction) :
        ErrorFunction<T, NeuralNetwork<T>>(baseFunction)
    {
        // Determine whether or not it is appropriate to use the softmax
        // optimization for simplified gradient caluclations. The dynamic cast
        // should return a null pointer if the last layer is not softmax.
        // We also make sure there are at least 2 layers.
        Layer<T>* outputLayer   = mBaseFunction.getOutputLayer();
        SoftmaxLayer<T>* ptr    = dynamic_cast<SoftmaxLayer<T>*>(outputLayer);
        mUseSoftmaxOptimization = (ptr != nullptr) && (mBaseFunction.getNumLayers() > 1);
    }

    T evaluate(const Matrix<T>& features, const Matrix<T>& labels)
    {
        // A very small number is added to the input to prevent log(0) becoming NaN.
        const T EPSILON = std::numeric_limits<T>::epsilon();
        const size_t N  = features.rows();
        const size_t M  = labels.cols();
        static vector<T> prediction(M);

        T sum{};
        for (size_t i = 0; i < N; ++i)
        {
            mBaseFunction.evaluate(features[i], prediction);

            const vector<T>& row = labels[i];
            for (size_t j = 0; j < M; ++j)
                sum += row[j] * std::log(prediction[j] + EPSILON);
        }

        return -sum;
    }

    void calculateGradientInputs(const Matrix<T>& features,
        const Matrix<T>& labels, vector<T>& gradient)
    {
        if (mUseSoftmaxOptimization)
            calculateGradientInputsOpt(features, labels, gradient);
        else calculateGradientInputsUnopt(features, labels, gradient);
    }

    void calculateGradientParameters(const Matrix<T>& features,
        const Matrix<T>& labels, vector<T>& gradient)
    {
        if (mUseSoftmaxOptimization)
            calculateGradientParametersOpt(features, labels, gradient);
        else calculateGradientParametersUnopt(features, labels, gradient);
    }

    void calculateHessianInputs(const Matrix<T>& features, const Matrix<T>& labels,
        Matrix<T>& hessian)
    {
        // TODO
    }

    void calculateHessianParameters(const Matrix<T>& features,
        const Matrix<T>& labels, Matrix<T>& hessian)
    {
        // TODO
    }

private:
    // When the last layer is a softmax, the gradient calculation process can
    // be simplified computationally. This flag controls whether
    // calculateGradientXXXOpt or calculateGradientXXXUnopt will be called.
    bool mUseSoftmaxOptimization;

    void calculateGradientInputsOpt(const Matrix<T>& features,
        const Matrix<T>& labels, vector<T>& gradient)
    {
        const size_t N    = mBaseFunction.getInputs();
        const size_t M    = mBaseFunction.getOutputs();
        const size_t rows = features.rows();

        std::fill(gradient.begin(), gradient.end(), T{});
        static vector<T> evaluation(M);
        static vector<T> tempGradient(N);

        // Calculate a partial gradient for each row in the training data
        for (size_t i = 0; i < rows; ++i)
        {
            const vector<T>& feature = features[i];
            const vector<T>& label   = labels[i];

            // Forward prop
            mBaseFunction.evaluate(feature, evaluation);

            // Calculate the deltas for each node in the network
            {
                int layer = mBaseFunction.getNumLayers() - 1;

                // Calculate the deltas on the last layer first. Since the last
                // layer is known to be a softmax, and the softmax layer
                // doesn't use its deltas for gradient calculation, we can
                // actually skip this step completely.
                // vector<T>& outputDeltas = mBaseFunction.getLayer(layer)->getDeltas();
                // for (size_t j = 0; j < M; ++j)
                //    outputDeltas[j] = -label[j] / evaluation[j];
                layer--;

                // Calculate the deltas for the layer preceeding the softmax layer
                // using the optimized approach (since we know what the answer
                // should be already).
                vector<T>& preDeltas = mBaseFunction.getLayer(layer)->getDeltas();
                for (size_t j = 0; j < M; ++j)
                    preDeltas[j] = evaluation[j] - label[j];

                // Calculate the remaining deltas like normal
                for (int i = layer; i >= 1; --i)
                {
                    Layer<T>* current = mBaseFunction.getLayer(i);
                    Layer<T>* prev    = mBaseFunction.getLayer(i - 1);

                    current->calculateDeltas(prev->getActivation(), prev->getDeltas());
                }
            }

            // Calculate the gradient based on the deltas. Values are summed
            // for each pattern.
            mBaseFunction.getLayer(0)->calculateDeltas(feature, tempGradient);
            vAdd(tempGradient.data(), gradient.data(), N);
        }

        // We also need to divide by the batch size to get an average gradient.
        vScale(gradient.data(), 1.0/rows, N);
    }

    void calculateGradientInputsUnopt(const Matrix<T>& features,
        const Matrix<T>& labels, vector<T>& gradient)
    {
        const size_t N    = mBaseFunction.getInputs();
        const size_t M    = mBaseFunction.getOutputs();
        const size_t rows = features.rows();

        std::fill(gradient.begin(), gradient.end(), T{});
        static vector<T> evaluation(M);
        static vector<T> tempGradient(N);

        // Calculate a partial gradient for each row in the training data
        for (size_t i = 0; i < rows; ++i)
        {
            const vector<T>& feature = features[i];
            const vector<T>& label   = labels[i];

            // Forward prop
            mBaseFunction.evaluate(feature, evaluation);

            // Calculate the deltas for each node in the network
            {
                // Calculate the deltas on the last layer first
                vector<T>& outputDeltas = mBaseFunction.getOutputLayer()->getDeltas();
                for (size_t j = 0; j < M; ++j)
                    outputDeltas[j] = -label[j] / evaluation[j];

                mBaseFunction.calculateDeltas();
            }

            // Calculate the gradient based on the deltas. Values are summed
            // for each pattern.
            mBaseFunction.getLayer(0)->calculateDeltas(feature, tempGradient);
            vAdd(tempGradient.data(), gradient.data(), N);
        }

        // We also need to divide by the batch size to get an average gradient.
        vScale(gradient.data(), 1.0/rows, N);
    }

    void calculateGradientParametersOpt(const Matrix<T>& features,
        const Matrix<T>& labels, vector<T>& gradient)
    {
        const size_t N    = mBaseFunction.getNumParameters();
        const size_t M    = mBaseFunction.getOutputs();
        const size_t rows = features.rows();

        std::fill(gradient.begin(), gradient.end(), T{});
        static vector<T> evaluation(M);

        // Calculate a partial gradient for each row in the training data
        for (size_t i = 0; i < rows; ++i)
        {
            const vector<T>& feature = features[i];
            const vector<T>& label   = labels[i];

            // Forward prop
            mBaseFunction.evaluate(feature, evaluation);

            // Calculate the deltas for each node in the network
            {
                int layer = mBaseFunction.getNumLayers() - 1;

                // Calculate the deltas on the last layer first. Since the last
                // layer is known to be a softmax, and the softmax layer
                // doesn't use its deltas for gradient calculation, we can
                // actually skip this step completely.
                //vector<T>& outputDeltas = mBaseFunction.getLayer(layer)->getDeltas();
                //for (size_t j = 0; j < M; ++j)
                //    outputDeltas[j] = -label[j] / evaluation[j];
                layer--;

                // Calculate the deltas for the layer preceeding the softmax layer
                // using the optimized approach (since we know what the answer
                // should be already).
                vector<T>& preDeltas = mBaseFunction.getLayer(layer)->getDeltas();
                for (size_t j = 0; j < M; ++j)
                    preDeltas[j] = evaluation[j] - label[j];

                // Calculate the remaining deltas like normal
                for (int i = layer; i >= 1; --i)
                {
                    Layer<T>* current = mBaseFunction.getLayer(i);
                    Layer<T>* prev    = mBaseFunction.getLayer(i - 1);

                    current->calculateDeltas(prev->getActivation(), prev->getDeltas());
                }
            }

            // Calculate the gradient based on the deltas. Values are summed
            // for each pattern.
            mBaseFunction.calculateGradientParameters(feature, gradient);
        }

        // We also need to divide by the batch size to get an average gradient.
        vScale(gradient.data(), 1.0/rows, N);
    }

    void calculateGradientParametersUnopt(const Matrix<T>& features,
        const Matrix<T>& labels, vector<T>& gradient)
    {
        const size_t N    = mBaseFunction.getNumParameters();
        const size_t M    = mBaseFunction.getOutputs();
        const size_t rows = features.rows();

        std::fill(gradient.begin(), gradient.end(), T{});
        static vector<T> evaluation(M);

        // Calculate a partial gradient for each row in the training data
        for (size_t i = 0; i < rows; ++i)
        {
            const vector<T>& feature = features[i];
            const vector<T>& label   = labels[i];

            // Forward prop
            mBaseFunction.evaluate(feature, evaluation);

            // Calculate the deltas for each node in the network
            {
                // Calculate the deltas on the last layer first
                vector<T>& outputDeltas = mBaseFunction.getOutputLayer()->getDeltas();
                for (size_t j = 0; j < M; ++j)
                    outputDeltas[j] = -label[j] / evaluation[j];

                mBaseFunction.calculateDeltas();
            }

            // Calculate the gradient based on the deltas. Values are summed
            // for each pattern.
            mBaseFunction.calculateGradientParameters(feature, gradient);
        }

        // We also need to divide by the batch size to get an average gradient.
        vScale(gradient.data(), 1.0/rows, N);
    }
};

};

#endif /* CROSSENTROPYFUNCTION_H */

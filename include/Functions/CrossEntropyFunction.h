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
        // TODO
    }

    void calculateGradientParameters(const Matrix<T>& features,
        const Matrix<T>& labels, vector<T>& gradient)
    {
        // TODO
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
        // TODO
    }

    void calculateGradientParameters(const Matrix<T>& features,
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

};

#endif /* CROSSENTROPYFUNCTION_H */

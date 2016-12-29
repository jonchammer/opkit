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
        // H(f(x, w)) = d/dw[-T/y(x, w)] * J(y(x, w)) +
        //              d/dw[J(y(x, w))] * (-T/y(x, w)), where
        // - H(f(x, w)) is what we're trying to calculate, the Hessian of the
        //   cross-entropy function with respect to the parameters (w)
        //   (size NxN)
        // - T is the label for the current sample (size M)
        // - y(x, w) is the output of the base function for the current sample
        //   (size M)
        // - J(y(x, w)) is the Jacobian of the base function (MxN)
        // - T/y(x, w) denotes an element-wise division (yielding a single vector
        //   of size M).
        //
        // d/dw[-T/y(x, w)] works out to be:
        //   [-t_i / (y_i^2)] * d/dw_j[y_i]
        // for each (i, j) in an MxN matrix. This doesn't turn out to be a clean
        // vector-matrix op, but it can be calculated naively. We basically
        // multiply each row of the model's Jacobian matrix by a constant term
        // that differs for each row.
        //
        // The "d/dw[J(y(x, w))] * (-T/y(x, w))" term is difficult to evaluate
        // correctly because the derivative of the base function's jacobian
        // matrix with respect to the parameters is actually an (M x N x N) 3D
        // tensor, where each of the M NxN slices is a Hessian matrix of the
        // model with respect to one of the M outputs of the base function.
        // Furthermore, multiplying by the (-T/y(x, w)) term, which is a 1 x M
        // matrix, only makes sense if the vector-matrix multiplication is
        // performed for each slice of the Hessian tensor, giving us N vector-
        // matrix multiplications (1 x M) * (M x N) ==> 1 x N. N * (1 x N)
        // produces a single N x N 2D matrix, which makes sense because the
        // Hessian of the cross-entropy function should have the same dimensions.

        // H = (T/Y^2) * J^T * J - (T/Y) * H_i
        const size_t N    = mBaseFunction.getNumParameters();
        const size_t M    = mBaseFunction.getOutputs();
        const size_t rows = features.rows();

        Matrix<T> jacobian;
        Matrix<T> jacobianWork;
        Matrix<T> localHessian;
        vector<T> evaluation(M);

        jacobian.setSize(M, N);
        jacobianWork.setSize(M, N);
        localHessian.setSize(N, N);

        hessian.setAll(T{});

        for (size_t i = 0; i < rows; ++i)
        {
            vector<T>& feature = features[i];
            vector<T>& label   = labels[i];

            // Calculate the model's Jacobian matrix
            mBaseFunction.calculateJacobianParameters(features[i], jacobian);
            jacobianWork.copyPart(jacobian, 0, 0, M, N);

            // Evaluate this sample
            if (mBaseFunction.cachesLastEvaluation())
                mBaseFunction.getLastEvaluation(evaluation);
            else mBaseFunction.evaluate(features[i], evaluation);

            // Calculate T/Y^2 * J
            for (size_t j = 0; j < M; ++j)
            {
                T c = label[j] / (evaluation[j] * evaluation[j]);
                for (size_t k = 0; k < N; ++k)
                    jacobianWork[j][k] *= c;
            }

            // Calculate T/Y^2 * J^T * J
            for (size_t j = 0; j < N; ++j)
            {
                for (size_t k = 0; k < N; ++k)
                {
                    for (size_t l = 0; l < M; ++l)
                        hessian[j][k] += jacobianWork[l][j] * jacobian[l][k];
                }
            }

            // Calculate T/Y^2 * J^T * J - (T/Y) * H_i
            vector<Matrix<T>> allLocalHessians(M);
            for (size_t j = 0; j < M; ++j)
            {
                allLocalHessians[j].setSize(N, N);
                mBaseFunction.calculateHessianParameters(features[i], j, allLocalHessians[j]);
            }

            for (size_t j = 0; j < N; ++j)
            {
                for (size_t k = 0; k < N; ++k)
                {
                    T sum{};
                    for (size_t l = 0; l < M; ++l)
                        sum += label[l] / evaluation[l] * jacobian[l][k];

                    hessian[j][k] -= sum;
                }
            }
        }
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

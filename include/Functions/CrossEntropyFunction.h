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
#include "Function.h"
#include "ErrorFunction.h"
#include "Dataset.h"
#include "Matrix.h"
#include "NeuralNetwork.h"
#include "Acceleration.h"
#include "PrettyPrinter.h"
using std::vector;

// The implementations of several of the key functions are placed
// here to avoid unnecessary code duplication.
namespace
{
    using opkit::Function;
    using opkit::Dataset;
    using opkit::Matrix;

    template <class T>
    T evaluate(Function<T>& baseFunction,
        const Dataset<T>& features, const Dataset<T>& labels)
    {
        // A very small number is added to the input to prevent log(0) becoming NaN.
        const T EPSILON = std::numeric_limits<T>::epsilon();
        const size_t N  = features.rows();
        const size_t M  = labels.cols();
        static vector<T> prediction(M);

        T sum{};
        for (size_t i = 0; i < N; ++i)
        {
            baseFunction.evaluate(features[i], prediction);

            const vector<T>& row = labels[i];
            for (size_t j = 0; j < M; ++j)
                sum += row[j] * std::log(prediction[j] + EPSILON);
        }

        return -sum;
    }

    template <class T>
    void calculateHessianInputs(Function<T>& baseFunction,
        const Dataset<T>& features, const Dataset<T>& labels,
        const Matrix<T>& hessian)
    {

    }

    template <class T>
    void calculateHessianParameters(Function<T>& baseFunction,
        const Dataset<T>& features, const Dataset<T>& labels,
        Matrix<T>& hessian)
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

        // H = (-T/Y^2) * J^T * J - (T/Y) * H_i
        const size_t N    = baseFunction.getNumParameters();
        const size_t M    = baseFunction.getOutputs();
        const size_t rows = features.rows();

        Matrix<T> jacobian(M, N);
        Matrix<T> jacobianWork(M, N);
        Matrix<T> localHessian(N, N);
        vector<T> evaluation(M);
        vector<Matrix<T>> allLocalHessians(M);
        Matrix<T> error1(1, M);
        Matrix<T> error2(1, M);

        hessian.fill(T{});

        for (size_t i = 0; i < rows; ++i)
        {
            const vector<T>& feature = features[i];
            const vector<T>& label   = labels[i];

            // Calculate the model's Jacobian Dataset
            baseFunction.calculateJacobianParameters(features[i], jacobian);

            // Evaluate this sample
            if (baseFunction.cachesLastEvaluation())
                baseFunction.getLastEvaluation(evaluation);
            else baseFunction.evaluate(features[i], evaluation);

            // Calculate the error vectors
            // e1 = -t1/y1, e2 = -t1/y1^2
            for (size_t j = 0; j < M; ++j)
            {
                T dividend   = T{1.0} / evaluation[j];
                T temp       = -label[j] * dividend;
                error1(0, j) = temp;
                error2(0, j) = temp * dividend;
            }

            // Calculate -T/Y^2 * J
            for (size_t j = 0; j < M; ++j)
            {
                T c = error2(0, j);
                for (size_t k = 0; k < N; ++k)
                    jacobianWork(j, k) = jacobian(j, k) * c;
            }

            // Calculate -T/Y^2 * J^T * J
            hessian += transpose(jacobianWork) * jacobian;

            // Calculate -T/Y^2 * J^T * J - (T/Y) * H_i
            for (size_t j = 0; j < M; ++j)
            {
                allLocalHessians[j].resize(N, N);
                baseFunction.calculateHessianParameters(features[i], j, allLocalHessians[j]);
            }

            for (size_t j = 0; j < N; ++j)
            {
                for (size_t k = 0; k < N; ++k)
                {
                    T sum{};
                    for (size_t l = 0; l < M; ++l)
                        sum += error1(0, l) * allLocalHessians[l](j, k);

                    hessian(j, k) -= sum;
                }
            }
        }
    }
}

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

    CrossEntropyFunction(Model& baseFunction) :
        ErrorFunction<T, Model>(baseFunction)
    {
        // Do nothing
    }

    T evaluate(const Dataset<T>& features, const Dataset<T>& labels)
    {
        return ::evaluate(mBaseFunction, features, labels);
    }

    void calculateGradientInputs(const Dataset<T>& features,
        const Dataset<T>& labels, vector<T>& gradient)
    {
        // When Cross-entropy is the error function, the gradient is
        // sum_i((-y_i / f(theta, x_i)) * J)
        // where J is the model's Jacobian, and i goes over the number of inputs
        const size_t N    = mBaseFunction.getInputs();
        const size_t M    = mBaseFunction.getOutputs();
        const size_t rows = features.rows();

        // Set the gradient to the zero vector
        std::fill(gradient.begin(), gradient.end(), T{});

        static Matrix<T> baseJacobian(M, N);
        static Matrix<T> error(1, M);
        static vector<T> evaluation(M);

        // The matrix 'grad' temporarily holds the contents of the gradient
        static Matrix<T> grad(1, N);
        grad.swap(gradient);

        for (size_t i = 0; i < rows; ++i)
        {
            // Calculate the Jacobian matrix of the base function at this point
            // with respect to the inputs
            mBaseFunction.calculateJacobianInputs(features[i], baseJacobian);

            // Calculate the error for this sample
            if (mBaseFunction.cachesLastEvaluation())
                mBaseFunction.getLastEvaluation(evaluation);
            else mBaseFunction.evaluate(features[i], evaluation);

            for (size_t j = 0; j < M; ++j)
                error(0, j) = -labels[i][j] / evaluation[j];

            grad += error * baseJacobian;
        }

        // Swap back so 'gradient' contains the correct information
        grad.swap(gradient);

        // Divide by the batch size to get the average gradient
        vScale(gradient.data(), 1.0/rows, N);
    }

    void calculateGradientParameters(const Dataset<T>& features,
        const Dataset<T>& labels, vector<T>& gradient)
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
        static Matrix<T> error(1, M);
        static vector<T> evaluation(M);

        // The matrix 'grad' temporarily holds the contents of the gradient
        static Matrix<T> grad(1, N);
        grad.swap(gradient);

        for (size_t i = 0; i < rows; ++i)
        {
            // Calculate the Jacobian matrix of the base function at this point
            // with respect to the model parameters
            mBaseFunction.calculateJacobianParameters(features[i], baseJacobian);

            // Calculate the error for this sample
            if (mBaseFunction.cachesLastEvaluation())
                mBaseFunction.getLastEvaluation(evaluation);
            else mBaseFunction.evaluate(features[i], evaluation);

            for (size_t j = 0; j < M; ++j)
                error(0, j) = -labels[i][j] / evaluation[j];

            grad += error * baseJacobian;
        }

        // Swap back so 'gradient' contains the correct information
        grad.swap(gradient);

        vScale(gradient.data(), 1.0/rows, N);
    }

    void calculateHessianInputs(const Dataset<T>& features,
        const Dataset<T>& labels, Matrix<T>& hessian)
    {
        return ::calculateHessianInputs(mBaseFunction,
            features, labels, hessian);
    }

    void calculateHessianParameters(const Dataset<T>& features,
        const Dataset<T>& labels, Matrix<T>& hessian)
    {
        ::calculateHessianParameters(mBaseFunction,
            features, labels, hessian);
    }
};

// Template specialization for Neural Networks, since there is a much more
// efficient mechanism for calculating the gradient with them.
template<class T>
class CrossEntropyFunction<T, NeuralNetwork<T>> :
    public ErrorFunction<T, NeuralNetwork<T>>
{
public:

    using ErrorFunction<T, NeuralNetwork<T>>::mBaseFunction;

    CrossEntropyFunction(NeuralNetwork<T>& baseFunction) :
        ErrorFunction<T, NeuralNetwork<T>>(baseFunction)
    {
        // Determine whether or not it is appropriate to use the softmax
        // optimization for simplified gradient calculations. The dynamic cast
        // should return a null pointer if the last layer is not a softmax.
        // We also make sure there are at least 2 layers.
        Layer<T>* outputLayer = mBaseFunction.getOutputLayer();
        SoftmaxLayer<T>* ptr  = dynamic_cast<SoftmaxLayer<T>*>(outputLayer);

        if ((ptr != nullptr) && (mBaseFunction.getNumLayers() > 1))
            mImp = new OptimizedImp(mBaseFunction);
        else mImp = new UnoptimizedImp(mBaseFunction);
    }

    ~CrossEntropyFunction()
    {
        delete mImp;
        mImp = nullptr;
    }

    T evaluate(const Dataset<T>& features, const Dataset<T>& labels)
    {
        return ::evaluate(mBaseFunction, features, labels);
    }

    void calculateGradientInputs(const Dataset<T>& features,
        const Dataset<T>& labels, vector<T>& gradient)
    {
        mImp->calculateGradientInputs(features, labels, gradient);
    }

    void calculateGradientParameters(const Dataset<T>& features,
        const Dataset<T>& labels, vector<T>& gradient)
    {
        mImp->calculateGradientParameters(features, labels, gradient);
    }

    void calculateHessianInputs(const Dataset<T>& features,
        const Dataset<T>& labels, Matrix<T>& hessian)
    {
        ::calculateHessianInputs(mBaseFunction,
            features, labels, hessian);
    }

    void calculateHessianParameters(const Dataset<T>& features,
        const Dataset<T>& labels, Matrix<T>& hessian)
    {
        ::calculateHessianParameters(mBaseFunction,
            features, labels, hessian);
    }

private:

    // The PIMPL idiom is used here. The implementation will be chosen in
    // the constructor, and all the normal calls are forwarded to that
    // implementation.
    struct Imp
    {
        Imp(NeuralNetwork<T>& baseFunction) :
            mBaseFunction(baseFunction) {};

        virtual void calculateGradientInputs(const Dataset<T>& features,
            const Dataset<T>& labels, vector<T>& gradient) = 0;
        virtual void calculateGradientParameters(const Dataset<T>& features,
            const Dataset<T>& labels, vector<T>& gradient) = 0;

        protected:
            NeuralNetwork<T>& mBaseFunction;
    };

    struct OptimizedImp : public Imp
    {
        OptimizedImp(NeuralNetwork<T>& baseFunction) :
            Imp(baseFunction) {}

        void calculateGradientInputs(const Dataset<T>& features,
            const Dataset<T>& labels, vector<T>& gradient) override
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

                        current->calculateDeltas(prev->getActivation(),
                            prev->getDeltas().data());
                    }
                }

                // Calculate the gradient based on the deltas. Values are summed
                // for each pattern.
                mBaseFunction.getLayer(0)->calculateDeltas(feature, tempGradient.data());
                vAdd(tempGradient.data(), gradient.data(), N);
            }

            // We also need to divide by the batch size to get an average gradient.
            vScale(gradient.data(), 1.0/rows, N);
        }

        void calculateGradientParameters(const Dataset<T>& features,
            const Dataset<T>& labels, vector<T>& gradient) override
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

                        current->calculateDeltas(prev->getActivation(),
                            prev->getDeltas().data());
                    }
                }

                // Calculate the gradient based on the deltas. Values are summed
                // for each pattern.
                mBaseFunction.calculateGradientParameters(feature, gradient.data());
            }

            // We also need to divide by the batch size to get an average gradient.
            vScale(gradient.data(), 1.0/rows, N);
        }
    };

    struct UnoptimizedImp : public Imp
    {
        UnoptimizedImp(NeuralNetwork<T>& baseFunction) :
            Imp(baseFunction) {}

        void calculateGradientInputs(const Dataset<T>& features,
            const Dataset<T>& labels, vector<T>& gradient) override
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
                mBaseFunction.getLayer(0)->calculateDeltas(feature, tempGradient.data());
                vAdd(tempGradient.data(), gradient.data(), N);
            }

            // We also need to divide by the batch size to get an average gradient.
            vScale(gradient.data(), 1.0/rows, N);
        }

        void calculateGradientParameters(const Dataset<T>& features,
            const Dataset<T>& labels, vector<T>& gradient) override
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
                mBaseFunction.calculateGradientParameters(feature, gradient.data());
            }

            // We also need to divide by the batch size to get an average gradient.
            vScale(gradient.data(), 1.0/rows, N);
        }
    };

    // The only piece of data for this class - a pointer to the chosen implementation
    Imp* mImp;
};

};

#endif /* CROSSENTROPYFUNCTION_H */

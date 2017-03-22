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
#include <cassert>
#include "Function.h"
#include "CostFunction.h"
#include "Dataset.h"
#include "Matrix.h"
#include "NeuralNetwork.h"
#include "SoftmaxLayer.h"
#include "Acceleration.h"
#include "PrettyPrinter.h"
using std::vector;

// The implementations of several of the key functions are placed
// here to avoid unnecessary code duplication.
namespace
{
    using opkit::Function;
    using opkit::Matrix;

    template <class T>
    void calculateHessianInputs(Function<T>& baseFunction,
        const Matrix<T>& features, const Matrix<T>& labels,
        const Matrix<T>& hessian)
    {
        // This function isn't implemented yet.
        assert(false);
    }

    template <class T>
    void calculateHessianParameters(Function<T>& baseFunction,
        const Matrix<T>& features, const Matrix<T>& labels,
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
        const size_t rows = features.getRows();

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
            // Calculate the model's Jacobian Dataset
            baseFunction.calculateJacobianParameters(features(i), jacobian);

            // Evaluate this sample
            baseFunction.evaluate(features(i), evaluation.data());

            // Calculate the error vectors
            // e1 = -t1/y1, e2 = -t1/y1^2
            for (size_t j = 0; j < M; ++j)
            {
                T dividend   = T{1.0} / evaluation[j];
                T temp       = -labels(i, j) * dividend;
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
                baseFunction.calculateHessianParameters(features(i), j, allLocalHessians[j]);
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
class CrossEntropyFunction : public CostFunction<T, Model>
{
public:

    using CostFunction<T, Model>::mBaseFunction;

    CrossEntropyFunction(Model& baseFunction) :
        CostFunction<T, Model>(baseFunction)
    {
        // Do nothing
    }

    T evaluate(const Matrix<T>& features, const Matrix<T>& labels)
    {
        // A very small number is added to the input to prevent log(0) becoming NaN.
        const T EPSILON = std::numeric_limits<T>::epsilon();
        const size_t N  = features.getRows();
        const size_t M  = labels.getCols();
        static vector<T> prediction(M);

        T sum{};
        for (size_t i = 0; i < N; ++i)
        {
            mBaseFunction.evaluate(features(i), prediction.data());
            for (size_t j = 0; j < M; ++j)
                sum += labels(i, j) * std::log(prediction[j] + EPSILON);
        }

        return -sum;
    }

    void calculateGradientInputs(const Matrix<T>& features,
        const Matrix<T>& labels, vector<T>& gradient)
    {
        // When Cross-entropy is the error function, the gradient is
        // sum_i((-y_i / f(theta, x_i)) * J)
        // where J is the model's Jacobian, and i goes over the number of inputs
        const size_t N    = mBaseFunction.getInputs();
        const size_t M    = mBaseFunction.getOutputs();
        const size_t rows = features.getRows();

        // Make sure gradient has enough space
        assert(gradient.size() >= N);

        // Set the gradient to the zero vector
        std::fill(gradient.begin(), gradient.end(), T{});

        static Matrix<T> baseJacobian(M, N);
        static Matrix<T> error(1, M);
        static vector<T> evaluation(M);

        // The matrix 'grad' temporarily holds the contents of the gradient
        Matrix<T> grad(gradient.data(), 1, N);

        for (size_t i = 0; i < rows; ++i)
        {
            // Calculate the Jacobian matrix of the base function at this point
            // with respect to the inputs
            mBaseFunction.calculateJacobianInputs(features(i), baseJacobian);

            // Calculate the error for this sample
            mBaseFunction.evaluate(features(i), evaluation.data());

            for (size_t j = 0; j < M; ++j)
                error(0, j) = -labels(i, j) / evaluation[j];

            grad += error * baseJacobian;
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
        const size_t rows = features.getRows();

        // Make sure gradient has enough space
        assert(gradient.size() >= N);

        // Set the gradient to the zero vector
        std::fill(gradient.begin(), gradient.end(), T{});

        static Matrix<T> baseJacobian;
        static Matrix<T> error(1, M);
        static vector<T> evaluation(M);

        // The matrix 'grad' temporarily holds the contents of the gradient
        Matrix<T> grad(gradient.data(), 1, N);

        for (size_t i = 0; i < rows; ++i)
        {
            // Calculate the Jacobian matrix of the base function at this point
            // with respect to the model parameters
            mBaseFunction.calculateJacobianParameters(features(i), baseJacobian);

            // Calculate the error for this sample
            mBaseFunction.evaluate(features(i), evaluation.data());

            for (size_t j = 0; j < M; ++j)
                error(0, j) = -labels(i, j) / evaluation[j];

            grad += error * baseJacobian;
        }

        vScale(gradient.data(), 1.0/rows, N);
    }

    void calculateHessianInputs(const Matrix<T>& features,
        const Matrix<T>& labels, Matrix<T>& hessian)
    {
        ::calculateHessianInputs(mBaseFunction,
            features, labels, hessian);
    }

    void calculateHessianParameters(const Matrix<T>& features,
        const Matrix<T>& labels, Matrix<T>& hessian)
    {
        ::calculateHessianParameters(mBaseFunction,
            features, labels, hessian);
    }
};

// Template specialization for Neural Networks, since there is a much more
// efficient mechanism for calculating the gradient with them.
template<class T>
class CrossEntropyFunction<T, NeuralNetwork<T>> :
    public CostFunction<T, NeuralNetwork<T>>
{
public:

    using CostFunction<T, NeuralNetwork<T>>::mBaseFunction;

    CrossEntropyFunction(NeuralNetwork<T>& baseFunction) :
        CostFunction<T, NeuralNetwork<T>>(baseFunction)
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

        // Calculate the error for most of the batches
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

    // Measures the Cross-entropy error for a single batch
    T evalBatch(Matrix<T>& batchFeatures, Matrix<T>& batchLabels,
        Matrix<T>& predictions)
    {
        const T EPSILON        = std::numeric_limits<T>::epsilon();
        const size_t batchSize = batchFeatures.getRows();
        const size_t N         = batchLabels.getCols();

        // Evaluate this minibatch
        mBaseFunction.evaluateBatch(batchFeatures, predictions);

        T sum {};
        for (size_t i = 0; i < batchSize; ++i)
        {
            for (size_t j = 0; j < N; ++j)
                 sum += batchLabels(i, j) * std::log(predictions(i, j) + EPSILON);
        }

        return -sum;
    }

public:

    void calculateGradientInputs(const Matrix<T>& features,
        const Matrix<T>& labels, vector<T>& gradient)
    {
        NeuralNetwork<T>& nn   = mBaseFunction;
        const size_t M         = nn.getOutputs();
        const size_t P         = nn.getInputs();
        const size_t batchSize = nn.getMaxBatchSize();

        // Make sure gradient has enough space
        assert(gradient.size() >= P);

        Matrix<T> batchFeatures((T*) features.data(), batchSize, P);
        Matrix<T> batchLabels((T*) labels.data(), batchSize, M);

        // Calculate gradients for most of the batches
        size_t rows = features.getRows();
        while (rows >= batchSize)
        {
            mImp->calculateGradientInputs(batchFeatures, batchLabels, gradient);

            // Move to the next batch
            T* featureData = batchFeatures.data();
            T* labelData   = batchLabels.data();
            batchFeatures.setData(featureData + batchSize * P);
            batchLabels.setData(labelData + batchSize * M);

            rows -= batchSize;
        }

        // Deal with the leftover elements
        if (rows > 0)
        {
            batchFeatures.reshape(rows, P);
            batchLabels.reshape(rows, M);
            mImp->calculateGradientInputs(batchFeatures, batchLabels, gradient);
        }

        // Divide by the number of rows to get the average gradient
        vScale(gradient.data(), T{1.0} / features.getRows(), P);
    }

    void calculateGradientParameters(const Matrix<T>& features,
        const Matrix<T>& labels, vector<T>& gradient)
    {
        NeuralNetwork<T>& nn   = mBaseFunction;
        const size_t N         = nn.getNumParameters();
        const size_t M         = nn.getOutputs();
        const size_t P         = nn.getInputs();
        const size_t batchSize = nn.getMaxBatchSize();

        // Make sure gradient has enough space
        assert(gradient.size() >= N);

        Matrix<T> batchFeatures((T*) features.data(), batchSize, P);
        Matrix<T> batchLabels((T*) labels.data(), batchSize, M);

        // Calculate gradients for most of the batches
        size_t rows = features.getRows();
        while (rows >= batchSize)
        {
            mImp->calculateGradientParameters(batchFeatures, batchLabels, gradient);

            // Move to the next batch
            T* featureData = batchFeatures.data();
            T* labelData   = batchLabels.data();
            batchFeatures.setData(featureData + batchSize * P);
            batchLabels.setData(labelData + batchSize * M);

            rows -= batchSize;
        }

        // Deal with the leftover elements
        if (rows > 0)
        {
            batchFeatures.reshape(rows, P);
            batchLabels.reshape(rows, M);
            mImp->calculateGradientParameters(batchFeatures, batchLabels, gradient);
        }

        // Divide by the number of rows to get the average gradient
        vScale(gradient.data(), T{1.0} / features.getRows(), N);
    }

    void calculateHessianInputs(const Matrix<T>& features,
        const Matrix<T>& labels, Matrix<T>& hessian)
    {
        ::calculateHessianInputs(mBaseFunction,
            features, labels, hessian);
    }

    void calculateHessianParameters(const Matrix<T>& features,
        const Matrix<T>& labels, Matrix<T>& hessian)
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

        virtual void calculateGradientInputs(const Matrix<T>& features,
            const Matrix<T>& labels, vector<T>& gradient) = 0;
        virtual void calculateGradientParameters(const Matrix<T>& features,
            const Matrix<T>& labels, vector<T>& gradient) = 0;

        protected:
            NeuralNetwork<T>& mBaseFunction;
    };

    struct OptimizedImp : public Imp
    {
        OptimizedImp(NeuralNetwork<T>& baseFunction) :
            Imp(baseFunction) {}

        void calculateGradientInputs(const Matrix<T>& batchFeatures,
            const Matrix<T>& batchLabels, vector<T>& gradient) override
        {
            NeuralNetwork<T>& nn   = mBaseFunction;
            const size_t batchSize = batchFeatures.getRows();
            const size_t N         = batchLabels.getCols();
            const size_t M         = batchFeatures.getCols();

            static Matrix<T> predictions(batchSize, N);
            static Matrix<T> localGradients(batchSize, M);
            predictions.reshape(batchSize, N);
            localGradients.reshape(batchSize, M);

            // Evaluate this minibatch
            nn.evaluateBatch(batchFeatures, predictions);

            // Calculate the deltas for each node in the network
            int layer = nn.getNumLayers() - 1;

            // Calculate the deltas on the last layer first. Since the last
            // layer is known to be a softmax, and the softmax layer
            // doesn't use its deltas for gradient calculation, we can
            // actually skip this step completely.
            layer--;

            // Calculate the deltas for the layer preceeding the softmax layer
            // using the optimized approach (since we know what the answer
            // should be already).
            Matrix<T>& preDeltas = nn.getDeltas(layer);
            for (size_t i = 0; i < rows; ++i)
            {
                for (size_t j = 0; j < N; ++j)
                    preDeltas(i, j) = predictions(i, j) - batchLabels(i, j);
            }
            nn.backpropInputsBatch(layer, 1);

            // Calculate the gradients based on the deltas.
            nn.getLayer(0)->backpropInputsBatch(batchFeatures,
                nn.getActivation(0), nn.getDeltas(0), localGradients);

            // Add the local gradients to the current gradient vector
            for (size_t i = 0; i < batchSize; ++i)
                vAdd(localGradients(i), gradient.data(), M);
        }

        void calculateGradientParameters(const Matrix<T>& batchFeatures,
            const Matrix<T>& batchLabels, vector<T>& gradient) override
        {
            const size_t batchSize = batchFeatures.getRows();
            const size_t N         = batchLabels.getCols();
            const size_t M         = nn.getNumParameters();

            static Matrix<T> predictions(batchSize, N);
            static Matrix<T> localGradients(batchSize, M);
            predictions.reshape(batchSize, N);
            localGradients.reshape(batchSize, M);

            // Evaluate this minibatch
            mBaseFunction.evaluateBatch(batchFeatures, predictions);

            // Calculate the deltas for each node in the network
            int layer = nn.getNumLayers() - 1;

            // Calculate the deltas on the last layer first. Since the last
            // layer is known to be a softmax, and the softmax layer
            // doesn't use its deltas for gradient calculation, we can
            // actually skip this step completely.
            layer--;

            // Calculate the deltas for the layer preceeding the softmax layer
            // using the optimized approach (since we know what the answer
            // should be already).
            Matrix<T>& preDeltas = nn.getDeltas(layer);
            for (size_t i = 0; i < rows; ++i)
            {
                for (size_t j = 0; j < N; ++j)
                    preDeltas(i, j) = evaluation(i, j) - batchLabels(i, j);
            }
            mBaseFunction.backpropInputsBatch(layer, 1);

            // Calculate the gradients based on the deltas.
            nn.backpropParametersBatch(batchFeatures, localGradients);

            // Add the local gradients to the current gradient vector
            for (size_t i = 0; i < batchSize; ++i)
                vAdd(localGradients(i), gradient.data(), M);
        }
    };

    struct UnoptimizedImp : public Imp
    {
        UnoptimizedImp(NeuralNetwork<T>& baseFunction) :
            Imp(baseFunction) {}

        void calculateGradientInputs(Matrix<T>& batchFeatures,
            Matrix<T>& batchLabels, vector<T>& gradient)
        {
            NeuralNetwork<T>& nn   = mBaseFunction;
            const size_t batchSize = batchFeatures.getRows();
            const size_t N         = batchLabels.getCols();
            const size_t M         = batchFeatures.getCols();

            static Matrix<T> predictions(batchSize, N);
            static Matrix<T> localGradients(batchSize, M);
            predictions.reshape(batchSize, N);
            localGradients.reshape(batchSize, M);

            // Evaluate this minibatch
            nn.evaluateBatch(batchFeatures, predictions);

            // Calculate the deltas for each node in the network
            Matrix<T>& outputDeltas = nn.getDeltas(nn.getNumLayers() - 1);
            for (size_t i = 0; i < rows; ++i)
            {
                for (size_t j = 0; j < N; ++j)
                    outputDeltas(i, j) = -batchLabels(i, j) / predictions(i, j);
            }
            nn.backpropInputsBatch();

            // Calculate the gradients based on the deltas.
            nn.getLayer(0)->backpropInputsBatch(batchFeatures,
                nn.getActivation(0), nn.getDeltas(0), localGradients);

            // Add the local gradients to the current gradient vector
            for (size_t i = 0; i < batchSize; ++i)
                vAdd(localGradients(i), gradient.data(), M);
        }

        void calculateGradientParameters(Matrix<T>& batchFeatures,
            Matrix<T>& batchLabels, vector<T>& gradient)
        {
            const size_t batchSize = batchFeatures.getRows();
            const size_t N         = batchLabels.getCols();
            const size_t M         = nn.getNumParameters();

            static Matrix<T> predictions(batchSize, N);
            static Matrix<T> localGradients(batchSize, M);
            predictions.reshape(batchSize, N);
            localGradients.reshape(batchSize, M);

            // Evaluate this minibatch
            mBaseFunction.evaluateBatch(batchFeatures, predictions);

            // Calculate the deltas for each node in the network
            Matrix<T>& outputDeltas = mBaseFunction.getDeltas(nn.getNumLayers() - 1);
            for (size_t i = 0; i < rows; ++i)
            {
                for (size_t j = 0; j < N; ++j)
                    outputDeltas(i, j) = -batchLabels(i, j) / predictions(i, j);
            }
            mBaseFunction.backpropInputsBatch();

            // Calculate the gradients based on the deltas.
            nn.backpropParametersBatch(features, localGradients);

            // Add the local gradients to the current gradient vector
            for (size_t i = 0; i < batchSize; ++i)
                vAdd(localGradients(i), gradient.data(), M);
        }
    };

    // The only piece of data for this class - a pointer to the chosen implementation
    Imp* mImp;
};

};

#endif /* CROSSENTROPYFUNCTION_H */

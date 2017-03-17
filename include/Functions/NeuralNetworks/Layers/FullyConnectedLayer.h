#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H

#include "Layer.h"
#include "Matrix.h"
#include "Acceleration.h"

namespace opkit
{

// Fully connected layers are the basis of the MLP. They perform the linear
// transformation y = W * x + b, where the W matrix and the b vector are
// optimizable parameters.
template <class T>
class FullyConnectedLayer : public Layer<T>
{
public:

    // Allows us to use the members in the base class without specifying
    // their complete names
    using Layer<T>::mParameters;
    using Layer<T>::mInputs;
    using Layer<T>::mOutputs;
    using Layer<T>::mDeltas;
    using Layer<T>::mActivation;
    using Layer<T>::mBatchSize;

    // Create a new FullyConnectedLayer. We need to specify the input and
    // output dimensions, as well as the maximum batch size.
    FullyConnectedLayer(const size_t inputs, const size_t outputs,
        const size_t batchSize) :
        Layer<T>(inputs, outputs, batchSize),
        mAverageMask(new T[batchSize])
    {
        std::fill(mAverageMask, mAverageMask + batchSize, T{1.0}/batchSize);
    }

    ~FullyConnectedLayer()
    {
        delete[] mAverageMask;
        mAverageMask = nullptr;
    }

    virtual void eval(const Matrix<T>& x) override
    {
        const T* xData  = x.data();
        T* yData        = mActivation.data();
        const size_t N  = mBatchSize;

        // y = x * W^T + b
        // Weights are arranged as an 'mOutputs' x 'mInputs' matrix in row-major
        // ordering, followed directly by the biases. E.g.:
        // [w11, w21, ... wn1]
        // [w12, w22, ... wn2]
        // [...  ...  ... ...]
        // [w1m, w2m, ... wnm]
        // [b1, b2,   ...  bm]
        mmtMultiply(xData, mParameters, yData, N, mOutputs, mInputs);

        // We could also multiply [1, 1, ...]^T * biases to get a full matrix
        // that could directly be added to y, but that would involve more
        // memory overhead.
        const T* biases = mParameters + (mInputs * mOutputs);
        for (size_t i = 0; i < N; ++i)
        {
            vAdd(biases, yData, mOutputs);
            yData += mOutputs;
        }
    }

    void calculateDeltas(const Matrix<T>& x, T* destination) override
    {
        const T* deltas = mDeltas.data();
        const size_t N  = mBatchSize;

        // Calculate destination = deltas * W
        mmMultiply(deltas, mParameters, destination, N, mInputs, mOutputs);
    }

    void calculateGradient(const Matrix<T>& x, T* gradient) override
    {
        const T* input  = x.data();
        const T* deltas = mDeltas.data();
        const size_t N  = mBatchSize;

        // Calculate the sum of the gradients for each sample in the batch using
        // a single matrix multiplication. We then need to divide every cell by
        // the batch size to get the average gradient. We use the formula:
        // gradient(weights) = (deltas^T * x) / N;
        mtmMultiply(deltas, input, gradient, mOutputs, mInputs, N, T{1.0} / N);

        // Generate the average bias gradient by taking the average of the deltas
        // across the columns (or equivalently, by taking the average across the
        // rows in the transpose of the deltas). We implement this by
        // multiplying: deltas^T * the vector [1/N, 1/N, ... ].
        mtvMultiply(deltas, mAverageMask, gradient + mInputs * mOutputs, N, mOutputs);
    }

    size_t getNumParameters() const override
    {
        // N * M for the weights matrix and M for the bias terms
        return mInputs * mOutputs + mOutputs;
    }

    std::string getName() const override
    {
        return "Fully Connected Layer";
    }

private:
    T* mAverageMask; // Contains [1 / batchSize, 1 / batchSize, ...]
};
}

#endif

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

    // Create a new FullyConnectedLayer. We need to specify the input and
    // output dimensions.
    FullyConnectedLayer(const size_t inputs, const size_t outputs) :
        Layer<T>(inputs, outputs) {}

    virtual void forwardSingle(const T* x, T* y) override
    {
        // y = W * x + b
        mvMultiply(mParameters, x, y, mOutputs, mInputs);
        vAdd(mParameters + (mInputs * mOutputs), y, mOutputs);
    }

    virtual void forwardBatch(const Matrix<T>& x, Matrix<T>& y) override
    {
        const T* xData  = x.data();
        T* yData        = y.data();
        const size_t N  = x.getRows();

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

    void backpropInputsSingle(const T* x, const T* y,
        const T* deltas, T* dest) override
    {
        // dest = W^T * deltas
        mtvMultiply(mParameters, deltas, dest, mOutputs, mInputs);
    }

    void backpropInputsBatch(const Matrix<T>& x, const Matrix<T>& y,
        const Matrix<T>& deltas, Matrix<T>& dest) override
    {
        const T* deltasData = deltas.data();
        T* destData         = dest.data();
        const size_t N      = x.getRows();

        // Calculate destination = deltas * W
        mmMultiply(deltasData, mParameters, destData, N, mInputs, mOutputs);
    }

    void backpropParametersSingle(const T* x, const T* deltas, T* dest) override
    {
        // dest_parameters = outer product(deltas, x)
        std::fill(dest, dest + mInputs*mOutputs, T{});
        outerProduct(deltas, x, dest, mOutputs, mInputs);

        // dest_biases = deltas
        vCopy(deltas, dest + (mOutputs * mInputs), mOutputs);
    }

    void backpropParametersBatch(const Matrix<T>& x, const Matrix<T>& deltas,
        T* dest) override
    {
        const size_t N = x.getRows();

        // Calculate the sum of the gradients for each sample in the batch using
        // a single matrix multiplication. We then need to divide every cell by
        // the batch size to get the average gradient. We use the formula:
        // gradient(weights) = (deltas^T * x) / N;
        mtmMultiply(deltas.data(), x.data(), dest, mOutputs, mInputs, N, T{1.0} / N);

        // Generate the average bias gradient by taking the average of the deltas
        // across the columns (or equivalently, by taking the average across the
        // rows in the transpose of the deltas). We implement this by
        // multiplying: deltas^T * the vector [1/N, 1/N, ... ].
        if (mOnes.getRows() != 1 || mOnes.getCols() != N)
        {
            mOnes.resize(1, N);
            mOnes.fill(T{1});
        }
        mtvMultiply(deltas.data(), mOnes.data(), dest + mInputs * mOutputs,
            N, mOutputs, T{1.0} / N);
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
    Matrix<T> mOnes;
};

}

#endif

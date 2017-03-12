#ifndef MASKED_SPARSE_LAYER_H
#define MASKED_SPARSE_LAYER_H

#include "Layer.h"
#include "FullyConnectedLayer.h"
#include "Matrix.h"
#include "Acceleration.h"
#include "Bitmask.h"

namespace opkit
{

// One of the SparseLayer implementations. This version applies a bitmask to the
// weights, setting many of them to 0.0 before performing the same operations as
// FullyConnectedLayer. This version will typically be more efficient for
// smaller layers, since accelerated Matrix ops can be used.
template <class T>
class MaskedSparseLayer : public FullyConnectedLayer<T>
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

    // Create a new MasekdSparseLayer. The user specifies how many inputs and
    // outputs this layer has, as well as which percentage of the connections
    // should be filled (between [0.0 and 1.0]). The given Rand object is used
    // to determine which connections are made.
    MaskedSparseLayer(const size_t inputs, const size_t outputs,
        const size_t batchSize, const double fillPercentage, Rand& rand) :
        FullyConnectedLayer<T>(inputs, outputs, batchSize), mMask(inputs * outputs)
    {
        // Effectively no mask when all cells are filled
        if (fillPercentage >= 1.0)
            mMask.setAll();

        // Mask a certain percentage of the connections
        else mMask.setRandom(rand, fillPercentage);
    }

    // Create a new MasekdSparseLayer. The user specifies how many inputs and
    // outputs this layer has, but by default, none of the connections are
    // enabled. Use getMask() to adjust which weights will be used.
    MaskedSparseLayer(const size_t inputs, const size_t outputs,
        const size_t batchSize) :
        FullyConnectedLayer<T>(inputs, outputs, batchSize), mMask(inputs * outputs)
    {}

    void eval(const Matrix<T>& x) override
    {
        const T* xData  = x.data();
        T* yData        = mActivation.data();
        const size_t N  = mBatchSize;

        // Apply the mask. This will zero out some portion of the weights so
        // they do not affect the computation. Note that if the weights are
        // zeroed here, we do not need to mask the weights again in
        // calculateDeltas(). The mask could be applied to the gradient that
        // is calculated in calcuateGradient(), but there's no need to do any
        // additional work.
        mMask.apply(mParameters);

        // y = x * W^T + b
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

    size_t getNumParameters() const override
    {
        // N * M for the weights matrix and M for the bias terms
        return mInputs * mOutputs + mOutputs;
    }

    Bitmask<T>& getMask()
    {
        return mMask;
    }

private:
    Bitmask<T> mMask;
};

}

#endif

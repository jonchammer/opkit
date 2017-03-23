#ifndef COMPRESSED_SPARSE_LAYER_H
#define COMPRESSED_SPARSE_LAYER_H

#include "Layer.h"
#include "Matrix.h"
#include "Acceleration.h"

namespace opkit
{

// Fundamentally, CompressedSparseLayer is similar to FullyConnectedLayer.
// The only difference is that this class uses sparse storage to reduce the
// computational complexity of the evaluation and backprop steps.
template <class T>
class CompressedSparseLayer : public Layer<T>
{
public:

    // Allows us to use the members in the base class without specifying
    // their complete names
    using Layer<T>::mParameters;
    using Layer<T>::mInputs;
    using Layer<T>::mOutputs;

    // Create a new MasekdSparseLayer. The user specifies how many inputs and
    // outputs this layer has, as well as which percentage of the connections
    // should be filled (between [0.0 and 1.0]). The given Rand object is used
    // to determine which connections are made.
    CompressedSparseLayer(const size_t inputs, const size_t outputs,
        const double fillPercentage, Rand& rand) :
        Layer<T>(inputs, outputs),
        mNumConnections(size_t(fillPercentage * inputs * outputs)),
        mWeights(nullptr, outputs, inputs)
    {
        // Assign rows and columns for each connection
        if (outputs * inputs == mNumConnections)
            mWeights.setAll();

        // Fill the connections randomly
        else mWeights.setRandom(rand, fillPercentage);
    }

    void forwardSingle(const T* x, T* y) override
    {
        mWeights.multiply(x, y);
        vAdd(mParameters + mNumConnections, y, mOutputs);
    }

    void backpropInputsSingle(const T* x, const T* y,
        const T* deltas, T* dest) override
    {
        // Calculate destination = W^T * deltas
        mWeights.multiplyTranspose(deltas, dest);
    }

    void backpropParametersSingle(const T* x, const T* deltas, T* dest) override
    {
        // gradient += deltas * x^T
        mWeights.outerProduct(deltas, x, dest);
        vAdd(deltas, dest + mNumConnections, mOutputs);
    }

    size_t getNumParameters() const override
    {
        return mNumConnections + mOutputs;
    }

    void onStorageAssigned() override
    {
        // As soon as this layer is assigned storage, we will need to sync the
        // parameters to the sparse weights matrix. (This might happen multiple
        // times as multiple layers are added to the network).
        mWeights.setData(mParameters);
    }

    // Simple getters
    size_t getNumConnections()           const { return mNumConnections; }
    SparseMatrixWrapper<T>& getWeights() const { return mWeights;        }

private:
    size_t mNumConnections;
    SparseMatrixWrapper<T> mWeights;
};

}

#endif

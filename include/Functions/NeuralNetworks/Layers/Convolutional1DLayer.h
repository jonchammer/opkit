#ifndef CONVOLUTIONAL_1D_LAYER_H
#define CONVOLUTIONAL_1D_LAYER_H

#include "Layer.h"
#include "Matrix.h"
#include "Acceleration.h"

namespace opkit
{

template <class T>
class Convolutional1DLayer : public Layer<T>
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

    Convolutional1DLayer
    (
        size_t inputSize, size_t batchSize, size_t inputChannels,
        size_t filterSize, size_t numFilters,
        size_t stride = 1, size_t zeroPadding = 0
    ) :

    // Superclass constructor - inputs, outputs, and batch size
    Layer<T>(inputSize * inputChannels,
        ((inputSize - filterSize + 2 * zeroPadding) / stride + 1) * numFilters,
        batchSize),

    // Paramaters
    mInputSize(inputSize),
    mInputChannels(inputChannels), mFilterSize(filterSize),
    mNumFilters(numFilters), mStride(stride), mZeroPadding(zeroPadding),
    mOutputSize((inputSize - filterSize + 2 * zeroPadding) / stride + 1),

    mInputMatrix(mOutputSize * batchSize, filterSize * inputChannels)
    {}

    ~Convolutional1DLayer()
    {}

    void eval(const Matrix<T>& x) override
    {
        // Expand x using the im2Row transformation. Each row of x will be
        // transformed into 'mOutputSize' rows according to the convolution
        // parameters. The relative order of the rows will be maintained.
        im2Row(x.data(), mInputSize, mBatchSize, mInputChannels, mFilterSize, 1,
            mZeroPadding, 0, mStride, 1, mInputMatrix.data());

        // Multiply the weights matrix by the transpose of the input matrix.
        // TODO: Check interlacing and (M, N, K) for this call.
        mtmMultiply(mParameters, mInputMatrix.data(), mActivation.data(),
            mInputMatrix.getCols(), mNumFilters, mInputMatrix.getRows() );

        // Add filter bias to each element
        // TODO: ...
    }

    void calculateDeltas(const Matrix<T>& x, T* destination) override
    {

    }

    void calculateGradient(const Matrix<T>& x, T* gradient) override
    {

    }

    size_t getNumParameters() const override
    {
        return (mFilterSize * mInputChannels + 1) * mNumFilters;
    }

private:
    size_t mInputSize;
    size_t mInputChannels;
    size_t mFilterSize;
    size_t mNumFilters;
    size_t mStride;
    size_t mZeroPadding;
    size_t mOutputSize;

    Matrix<T> mInputMatrix;
};

}

#endif

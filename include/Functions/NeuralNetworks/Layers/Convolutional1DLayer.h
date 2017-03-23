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
        static Matrix<T> tempActivation(mActivation.getRows(), mActivation.getCols());

        // Expand x using the im2Row transformation. Each row of x will be
        // transformed into 'mOutputSize' rows according to the convolution
        // parameters. The relative order of the rows will be maintained.
        im2Row(x.data(), mInputSize, mBatchSize, mInputChannels, mFilterSize, 1,
            mZeroPadding, 0, mStride, 1, mInputMatrix.data());

        // Multiply the weights matrix by the transpose of the input matrix.
        // activation = w * transpose(im2Row(x))
        mmtMultiply(mParameters, mInputMatrix.data(), tempActivation.data(),
            mNumFilters, mFilterSize * mInputChannels, mInputMatrix.getRows());

        // Perform a blockwise transpose using im2Row
        im2Row(tempActivation.data(), mOutputSize * mNumFilters, 1, mBatchSize,
            mOutputSize, 1, 0, 0, mOutputSize, 1, mActivation.data());

        // Add filter bias to each element
        const T* biases = mParameters + mFilterSize * mInputChannels * mNumFilters;
        for (size_t y = 0; y < mNumFilters; ++y)
        {
            for (size_t x = 0; x < mInputMatrix.getRows(); ++x)
                mActivation(y, x) += biases[y];
        }
    }

    void calculateDeltas(const Matrix<T>& x, T* destination) override
    {
        // TODO: Implement.
        // destination = crossCorrelation(deltas, weights)
    }

    void calculateGradient(const Matrix<T>& x, T* gradient) override
    {
        // We assume that im2Row has already been called using x, and the
        // results are stored in mInputMatrix.

        static Matrix<T> tempDeltas(mNumFilters, mOutputSize * mBatchSize);

        // Perform a blockwise transpose of the deltas using im2Row
        im2Row(mDeltas.data(), mOutputSize * mNumFilters, 1, mBatchSize,
            mOutputSize, 1, 0, 0, mOutputSize, 1, tempDeltas.data());

        // Calculate the sum of the gradients of the samples
        mmMultiply(tempDeltas.data(), mInputMatrix.data(), gradient,
            mNumFilters, mFilterSize * mInputChannels, mOutputSize * mBatchSize);

        // Divide by the batch size to get the average gradient
        vScale(gradient, T{1.0}/mBatchSize, getNumParameters());

        // The gradient for the biases is the average delta values
        T* gradBiases = gradient + (mFilterSize * mInputChannels * mNumFilters);
        std::fill(gradBiases, gradBiases + mNumFilters, T{});
        for (size_t y = 0; y < mDeltas.getRows(); ++y)
        {
            for (size_t x = 0; x < mDeltas.getCols(); ++x)
                gradBiases[x] += mDeltas(y, x);
        }
        T invN = T{1.0} / mDeltas.getRows();
        std::for_each(gradBiases, gradBiases + mNumFilters, [&invN](T& elem)
        {
            elem *= invN;
        });
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

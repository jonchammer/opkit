#ifndef CONVOLUTIONAL_1D_LAYER_H
#define CONVOLUTIONAL_1D_LAYER_H

#include "Layer.h"
#include "Matrix.h"
#include "Acceleration.h"

namespace opkit
{

// This is an implementation of a 1D convolutional layer. It supports multiple
// convolution kernels, as well as arbitrary zero padding and stride values.
// The implementation is based off of this paper:
// http://cs.nju.edu.cn/wujx/paper/CNN.pdf
// It uses the im2Row and row2Im transforms to reduce the convolutions to simple
// matrix multiplications for improved performance.
template <class T>
class Convolutional1DLayer : public Layer<T>
{
public:

    // Allows us to use the members in the base class without specifying
    // their complete names
    using Layer<T>::mParameters;

    Convolutional1DLayer
    (
        size_t inputSize, size_t inputChannels,
        size_t filterSize, size_t numFilters,
        size_t stride = 1, size_t zeroPadding = 0
    ) :
        // Superclass constructor - inputs, outputs
        Layer<T>(inputSize * inputChannels,
            ((inputSize - filterSize + 2 * zeroPadding) / stride + 1) * numFilters),

        // Paramaters
        mInputSize(inputSize),
        mInputChannels(inputChannels), mFilterSize(filterSize),
        mNumFilters(numFilters), mStride(stride), mZeroPadding(zeroPadding),
        mOutputSize((inputSize - filterSize + 2 * zeroPadding) / stride + 1),

        mInputMatrix(mOutputSize, filterSize * inputChannels),
        mIntermediateMatrix(mOutputSize, mFilterSize * mInputChannels)
    {}

    void forwardSingle(const T* x, T* y) override
    {
        // Expand x using the im2Row transformation. 'x' will be transformed
        // into 'mOutputSize' rows according to the convolution parameters.
        im2Row(x, mInputSize, 1, mInputChannels, mFilterSize, 1, mZeroPadding, 0,
            mStride, 1, mInputMatrix.data());

        // Multiply the weights matrix by the transpose of the input matrix.
        // y = w * transpose(im2Row(x))
        mmtMultiply(mParameters, mInputMatrix.data(), y,
            mNumFilters, mOutputSize, mFilterSize * mInputChannels);

        // Add the bias for each filter
        const T* biases = mParameters +
            (mFilterSize * mInputChannels * mNumFilters);

        for (size_t row = 0; row < mNumFilters; ++row)
        {
            const T bias = biases[row];
            for (size_t col = 0; col < mOutputSize; ++col)
                y[row * mOutputSize + col] += bias;
        }
    }

    void backpropInputsSingle(const T* x, const T* y, const T* deltas, T* dest) override
    {
        // destination = row2im(deltas^T * weights)
        // - deltas^T:           outputSize x numFilters
        // - weights:            numFilters x (filterSize * channels)
        // - deltas^T * weights: outputSize x (filterSize * channels)
        // destination:          channels x size
        mtmMultiply(deltas, mParameters, mIntermediateMatrix.data(),
            mOutputSize, mFilterSize * mInputChannels, mNumFilters);

        row2Im(mIntermediateMatrix.data(),
            mFilterSize, 1, mInputChannels,
            mInputSize, 1,
            mZeroPadding, 0, mStride, 1, dest);
    }

    void backpropParametersSingle(const T* x, const T* deltas, T* dest) override
    {
        // We assume that im2Row has already been called using x, and the
        // results are stored in mInputMatrix.

        // gradient_weights = deltas * im2Row(x)
        mmMultiply(deltas, mInputMatrix.data(), dest,
            mNumFilters, mFilterSize * mInputChannels, mOutputSize);

        // gradient_biases = sum_per_row(deltas)
        static Matrix<T> ones(mOutputSize, 1, T{1});
        mvMultiply(deltas, ones.data(),
            dest + mFilterSize * mInputChannels * mNumFilters,
            mNumFilters, mOutputSize);
    }

    size_t getNumParameters() const override
    {
        return (mFilterSize * mInputChannels + 1) * mNumFilters;
    }

    std::string getName() const
    {
        return "1D Convolutional Layer";
    }

    std::string* getProperties(size_t& numElements) const override
    {
        std::string* arr = new std::string[5];

        char buffer[1024];
        snprintf(buffer, 1024, "(%zux%zu) -> (%zux%zu)",
            mInputSize, mInputChannels,
            mOutputSize, mNumFilters);
        arr[0] = string(buffer);

        snprintf(buffer, 1024, "%-12s (%zux%zu)", "Filter Size:",
            mFilterSize, mInputChannels);
        arr[1] = string(buffer);

        snprintf(buffer, 1024, "%-12s %zu", "Num Filters:", mNumFilters);
        arr[2] = string(buffer);

        snprintf(buffer, 1024, "%-12s %zu", "Stride:", mStride);
        arr[3] = string(buffer);

        snprintf(buffer, 1024, "%-12s %zu", "Padding:", mZeroPadding);
        arr[4] = string(buffer);

        numElements = 5;
        return arr;
    }

    size_t getOutputSize() const     { return mOutputSize; }
    size_t getOutputChannels() const { return mNumFilters; }

private:
    size_t mInputSize;
    size_t mInputChannels;
    size_t mFilterSize;
    size_t mNumFilters;
    size_t mStride;
    size_t mZeroPadding;
    size_t mOutputSize;

    Matrix<T> mInputMatrix;
    Matrix<T> mIntermediateMatrix;
};

}

#endif

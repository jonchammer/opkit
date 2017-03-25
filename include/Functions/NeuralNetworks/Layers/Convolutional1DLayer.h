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
        size_t zeroPadding = 0, size_t stride = 1
    ) :
        // Superclass constructor - inputs, outputs
        Layer<T>(inputSize * inputChannels,
            ((inputSize - filterSize + 2 * zeroPadding) / stride + 1) * numFilters),

        // Paramaters
        mInputSize(inputSize), mInputChannels(inputChannels),
        mFilterSize(filterSize), mNumFilters(numFilters),
        mZeroPadding(zeroPadding), mStride(stride),
        mOutputSize((inputSize - filterSize + 2 * zeroPadding) / stride + 1),

        mInputMatrix(filterSize * inputChannels, mOutputSize),
        mIntermediateMatrix(mFilterSize * mInputChannels, mOutputSize)
    {}

    void forwardSingle(const T* x, T* y) override
    {
        // Expand x using the im2col transformation. 'x' will be transformed
        // into 'mOutputSize' columns according to the convolution parameters.
        im2col(x, mInputSize, 1, mInputChannels, mFilterSize, 1, mZeroPadding, 0,
            mStride, 1, 1, 1, mInputMatrix.data());

        // Multiply the weights matrix by the input matrix.
        // y = w * im2col(x)
        mmMultiply(mParameters, mInputMatrix.data(), y,
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
        // destination = col2im(weights^T * deltas)
        // - weights^T:          (filterSize * channels) x numFilters
        // - deltas:             numFilters x outputSize
        // - weights^T * deltas: (filterSize * channels) x outputSize
        // destination:          size x channels
        mtmMultiply(mParameters, deltas, mIntermediateMatrix.data(),
            mFilterSize * mInputChannels, mOutputSize, mNumFilters);

        col2im(mIntermediateMatrix.data(),
            mInputSize, 1, mInputChannels, mFilterSize, 1, mZeroPadding, 0,
            mStride, 1, 1, 1, dest);
    }

    void backpropParametersSingle(const T* x, const T* deltas, T* dest) override
    {
        im2col(x, mInputSize, 1, mInputChannels, mFilterSize, 1, mZeroPadding, 0,
            mStride, 1, 1, 1, mInputMatrix.data());

        // gradient_weights = deltas * transpose(im2col(x))
        mmtMultiply(deltas, mInputMatrix.data(), dest,
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

        snprintf(buffer, 1024, "%-12s %zu", "Padding:", mZeroPadding);
        arr[3] = string(buffer);

        snprintf(buffer, 1024, "%-12s %zu", "Stride:", mStride);
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
    size_t mZeroPadding;
    size_t mStride;
    size_t mOutputSize;

    Matrix<T> mInputMatrix;
    Matrix<T> mIntermediateMatrix;
};

}

#endif

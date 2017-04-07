#ifndef REGULARIZED_1D_CONVOLUTION_LAYER_H
#define REGULARIZED_1D_CONVOLUTION_LAYER_H

#include "FullyConnectedLayer.h"

namespace opkit
{

template <class T>
class Regularized1DConvolutionLayer : public FullyConnectedLayer<T>
{
public:

    // Allows us to use the members in the base class without specifying
    // their complete names
    using Layer<T>::mParameters;
    using Layer<T>::mInputs;
    using Layer<T>::mOutputs;

    constexpr static T DEFAULT_ALPHA = 0.001;
    constexpr static T DEFAULT_BETA  = 0.0001;

    Regularized1DConvolutionLayer
    (
        size_t inputSize, /*size_t inputChannels,*/
        size_t filterSize, /*size_t numFilters,*/
        size_t zeroPadding = 0, size_t stride = 1,
        T alpha = DEFAULT_ALPHA, T beta = DEFAULT_BETA
    ) :
        // Superclass constructor - inputs, outputs
        FullyConnectedLayer<T>(inputSize/* * inputChannels*/,
            ((inputSize - filterSize + 2 * zeroPadding) / stride + 1)/* * numFilters*/),

        // Paramaters
        mInputSize(inputSize),     /*mInputChannels(inputChannels),*/
        mFilterSize(filterSize),   /*mNumFilters(numFilters),*/
        mZeroPadding(zeroPadding), mStride(stride),
        mOutputSize((inputSize - filterSize + 2 * zeroPadding) / stride + 1),
        mAlpha(alpha), mBeta(beta),

        mWindowedParameters(mInputSize * mOutputSize),
        mShareBiases(true)
    {
        // Determine which cells are part of the windows
        for (size_t i = 0; i < mFilterSize; ++i)
        {
            for (int y = 0; y < mOutputSize; ++y)
            {
                int x = mStride * y - mZeroPadding + i;
                if (x >= 0 && x < mInputSize)
                    mWindowedParameters[y * mInputSize + x] = true;
            }
        }
    }

    void backpropParametersSingle(const T* x, const T* deltas, T* dest) override
    {
        FullyConnectedLayer<T>::backpropParametersSingle(x, deltas, dest);

        // Add regularization constraint to normal gradient
        updateGradient(dest);
    }

    void backpropParametersBatch(const Matrix<T>& x, const Matrix<T>& deltas,
        T* dest) override
    {
        //std::fill(dest, dest + mInputs * mOutputs + mOutputs, T{});
        FullyConnectedLayer<T>::backpropParametersBatch(x, deltas, dest);

        // Add regularization constraint to normal gradient
        updateGradient(dest);
    }

    void reportStats()
    {
        // Mean of each cell in the kernel
        vector<T> means(mFilterSize + 1);
        for (size_t i = 0; i < mFilterSize; ++i)
        {
            size_t count = 0;
            for (int y = 0; y < mOutputSize; ++y)
            {
                int x = mStride * y - mZeroPadding + i;
                if (x >= 0 && x < mInputSize)
                {
                    means[i] += mParameters[y * mInputSize + x];
                    ++count;
                }
            }
            means[i] /= count;
            std::cout << "Mean (window) " << i << " = " << means[i] << std::endl;
        }

        // Variance of each kernel
        for (size_t i = 0; i < mFilterSize; ++i)
        {
            T sum{};
            size_t count = 0;
            for (int y = 0; y < mOutputSize; ++y)
            {
                int x = mStride * y - mZeroPadding + i;
                if (x >= 0 && x < mInputSize)
                {
                    T dx = (mParameters[y * mInputSize + x] - means[i]);
                    sum += dx * dx;
                    ++count;
                }
            }

            std::cout << "Variance (window)" << i << " = " << sum / count << std::endl;
        }

        // Mean of cells outside the kernel
        T sum{};
        size_t count = 0;
        for (size_t i = 0; i < mInputSize * mOutputSize; ++i)
        {
            if (!mWindowedParameters[i])
            {
                sum += mParameters[i];
                ++count;
            }
        }
        T mean = sum / count;
        std::cout << "Mean (outside) = " << mean << std::endl;

        // Variance of the cells outside the kernel
        sum = T{};
        for (size_t i = 0; i < mInputSize * mOutputSize; ++i)
        {
            if (!mWindowedParameters[i])
            {
                T dx = mParameters[i] - mean;
                sum += dx * dx;
            }
        }
        std::cout << "Variance (outside) = " << sum / count << std::endl;

        // Mean of bias terms
        T* biases = mParameters + mOutputSize * mInputSize;
        for (int y = 0; y < mOutputSize; ++y)
            means[mFilterSize] += biases[y];
        T biasMean = means[mFilterSize] / mOutputSize;
        std::cout << "Mean (bias) = " << biasMean << std::endl;

        // Variance of bias terms
        sum = T{};
        for (int y = 0; y < mOutputSize; ++y)
        {
            T dx = biases[y] - biasMean;
            sum += dx * dx;
        }
        std::cout << "Variance (bias) = " << sum / mOutputSize << std::endl;
    }

    std::string getName() const override
    {
        return "Regularized 1D Convolutional Layer";
    }

private:
    size_t mInputSize;
    //size_t mInputChannels;
    size_t mFilterSize;
    //size_t mNumFilters;
    size_t mZeroPadding;
    size_t mStride;
    size_t mOutputSize;

    T mAlpha, mBeta;
    std::vector<bool> mWindowedParameters;
    bool mShareBiases;

    void updateGradient(T* gradient)
    {
        // Calculate the appropriate means
        vector<T> means(mFilterSize + 1);
        for (size_t i = 0; i < mFilterSize; ++i)
        {
            size_t count = 0;
            for (int y = 0; y < mOutputSize; ++y)
            {
                int x = mStride * y - mZeroPadding + i;
                if (x >= 0 && x < mInputSize)
                {
                    means[i] += mParameters[y * mInputSize + x];
                    ++count;
                }
            }
            means[i] /= count;
        }

        if (mShareBiases)
        {
            T* biases = mParameters + mOutputSize * mInputSize;
            for (int y = 0; y < mOutputSize; ++y)
                means[mFilterSize] += biases[y];
            means[mFilterSize] /= mOutputSize;
        }

        // Update the gradient for the cells in the window
        for (size_t i = 0; i < mFilterSize; ++i)
        {
            for (int y = 0; y < mOutputSize; ++y)
            {
                int x = mStride * y - mZeroPadding + i;
                if (x >= 0 && x < mInputSize)
                {
                    gradient[y * mInputSize + x] -=
                        mAlpha * (mParameters[y * mInputSize + x] - means[i]);
                }
            }
        }

        // Update the gradient for the cells outside the window
        for (size_t i = 0; i < mInputSize * mOutputSize; ++i)
        {
            if (!mWindowedParameters[i])
                gradient[i] -= mBeta * sign(mParameters[i]);
        }

        // Update the gradient for the bias terms
        if (mShareBiases)
        {
            T* biases       = mParameters + mOutputSize * mInputSize;
            T* biasGradient = gradient + mOutputSize * mInputSize;
            for (int y = 0; y < mOutputSize; ++y)
                biasGradient[y] -= mAlpha * (biases[y] - means[mFilterSize]);
        }
    }
};

}

#endif

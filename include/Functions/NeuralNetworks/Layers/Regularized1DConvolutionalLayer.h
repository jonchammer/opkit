#ifndef REGULARIZED_1D_CONVOLUTION_LAYER_H
#define REGULARIZED_1D_CONVOLUTION_LAYER_H

#include "FullyConnectedLayer.h"
#include "VectorOps.h"

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
        size_t filterSize, size_t numFilters,
        size_t zeroPadding = 0, size_t stride = 1,
        T alpha = DEFAULT_ALPHA, T beta = DEFAULT_BETA
    ) :
        // Superclass constructor - inputs, outputs
        FullyConnectedLayer<T>(inputSize/* * inputChannels*/,
            ((inputSize - filterSize + 2 * zeroPadding) / stride + 1) * numFilters),

        // Paramaters
        mInputSize(inputSize),     /*mInputChannels(inputChannels),*/
        mFilterSize(filterSize),   mNumFilters(numFilters),
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

        T gradMagnitude = magnitude(dest, mInputs * mOutputs + mOutputs);
        if (gradMagnitude != T{})
            vScale(dest, (T{1.0} - mAlpha - mBeta) / gradMagnitude, mInputs * mOutputs + mOutputs);

        // Add regularization constraint to normal gradient
        updateGradient(dest);
    }

    void backpropParametersBatch(const Matrix<T>& x, const Matrix<T>& deltas,
        T* dest) override
    {
        FullyConnectedLayer<T>::backpropParametersBatch(x, deltas, dest);

        T gradMagnitude = magnitude(dest, mInputs * mOutputs + mOutputs);
        if (gradMagnitude != T{})
            vScale(dest, (T{1.0} - mAlpha - mBeta) / gradMagnitude, mInputs * mOutputs + mOutputs);

        // Add regularization constraint to normal gradient
        updateGradient(dest);
    }

    // TODO: UPDATE FOR NUM_FILTERS
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

    std::string* getProperties(size_t& numElements) const override
    {
        std::string* arr = new std::string[7];

        char buffer[1024];
        snprintf(buffer, 1024, "(%zux%zux1) -> (%zux%zux%zu)",
            mInputSize, 1,
            mOutputSize, 1, mNumFilters);
        arr[0] = string(buffer);

        snprintf(buffer, 1024, "%-12s (%zux%zu)", "Filter Size:",
            mFilterSize, 1);
        arr[1] = string(buffer);

        snprintf(buffer, 1024, "%-12s %zu", "Num Filters:", mNumFilters);
        arr[2] = string(buffer);

        snprintf(buffer, 1024, "%-12s %zu", "Padding:", mZeroPadding);
        arr[3] = string(buffer);

        snprintf(buffer, 1024, "%-12s %zu", "Stride:", mStride);
        arr[4] = string(buffer);

        snprintf(buffer, 1024, "%-12s %f", "Alpha:", mAlpha);
        arr[5] = string(buffer);

        snprintf(buffer, 1024, "%-12s %f", "Beta:", mBeta);
        arr[6] = string(buffer);

        numElements = 7;
        return arr;
    }

private:
    size_t mInputSize;
    //size_t mInputChannels;
    size_t mFilterSize;
    size_t mNumFilters;
    size_t mZeroPadding;
    size_t mStride;
    size_t mOutputSize;

    T mAlpha, mBeta;
    std::vector<bool> mWindowedParameters;
    bool mShareBiases;

    // For all weights inside the convolutional window, we minimize variance
    // along the major diagonals so the corresponding value from each filter
    // converges to a single value.
    void calculateGradientWindow(const T* weights, T* gradient)
    {
        // Calculate the appropriate means
        vector<T> means(mFilterSize);
        vector<T> counts(mFilterSize);
        for (size_t i = 0; i < mFilterSize; ++i)
        {
            size_t count = 0;
            for (int y = 0; y < mOutputSize; ++y)
            {
                int x = mStride * y - mZeroPadding + i;
                if (x >= 0 && x < mInputSize)
                {
                    means[i] += weights[y * mInputSize + x];
                    ++count;
                }
            }
            means[i] /= count;
            counts[i] = count;
        }

        // Update the gradient for the cells in the window
        for (size_t i = 0; i < mFilterSize; ++i)
        {
            for (int y = 0; y < mOutputSize; ++y)
            {
                int x = mStride * y - mZeroPadding + i;
                if (x >= 0 && x < mInputSize)
                {
                    gradient[y * mInputSize + x] =
                        (T{2} / T{counts[i]}) *
                            (weights[y * mInputSize + x] - means[i]);
                }
            }
        }
    }

    // For all of the weights outside the convolutional window, we apply L1
    // regularization to enduce sparcity (driving most weights to 0).
    void calculateGradientOutsideWindow(const T* weights, T* gradient)
    {
        // Update the gradient for the cells outside the window
        for (size_t i = 0; i < mInputSize * mOutputSize; ++i)
        {
            // Returns the sign of the argument without using a branch
            auto signFn = [](const T& val)
            {
                return (T(0) < val) - (val < T(0));
            };

            if (!mWindowedParameters[i])
                gradient[i] = signFn(weights[i]);
        }
    }

    // The biases are regularized as if they were part of the convolutional
    // window. The goal is for the variance between bias terms to be very small.
    void calculateGradientBiases(const T* biases, T* gradient)
    {
        T mean{};
        for (int y = 0; y < mOutputSize; ++y)
            mean += biases[y];
        mean /= mOutputSize;

        for (int y = 0; y < mOutputSize; ++y)
            gradient[y] = (T{2} / mOutputSize) * (biases[y] - mean);
    }

    void updateGradient(T* gradient)
    {
        const size_t N = mInputSize * mOutputSize;

        vector<T> windowGradient(N);
        vector<T> outsideWindowGradient(N);
        vector<T> biasGradient(mOutputSize);

        const T* weights = mParameters;
        const T* biases  = mParameters + mInputs * mOutputs;

        T* dest       = gradient;
        T* destBiases = gradient + mInputs * mOutputs;

        for (size_t filter = 0; filter < mNumFilters; ++filter)
        {
            // Calculate and normalize the gradient within the window
            calculateGradientWindow(weights, windowGradient.data());
            T windowMag = magnitude(windowGradient.data(), N);
            if (windowMag != T{})
                vAdd(windowGradient.data(), dest, N, mAlpha / windowMag);

            // Calculate and normalize the gradient outside the window
            calculateGradientOutsideWindow(weights, outsideWindowGradient.data());
            T outsideWindowMag = magnitude(outsideWindowGradient.data(), N);
            if (outsideWindowMag != T{})
                vAdd(outsideWindowGradient.data(), dest, N, mBeta / outsideWindowMag);

            // Calculate and normalize the graident for the bias terms.
            if (mShareBiases)
            {
                calculateGradientBiases(biases, biasGradient.data());
                T biasMag = magnitude(biasGradient.data(), mOutputSize);
                vAdd(biasGradient.data(), destBiases, mOutputSize, mAlpha / biasMag);
            }

            // Prepare for the next iteration
            weights    += mInputSize * mOutputSize;
            biases     += mOutputSize;
            dest       += mInputSize * mOutputSize;
            destBiases += mOutputSize;
        }
    }
};

}

#endif

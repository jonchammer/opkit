#ifndef RANDOM_CROP_LAYER_H
#define RANDOM_CROP_LAYER_H

#include <cassert>
#include "Layer.h"

namespace opkit
{

// This is a preprocessing layer that takes a muli-channel 2D image as input
// and produces a cropped version of the same image as ouput. A random
// cropping window is used during training, and the center of the image is
// used during testing. This layer has no learned parameters, so it will
// not contribute to the gradient, but it should preceed all learning layers
// in the neural network.
template <class T>
class RandomCropLayer : public Layer<T>
{
public:

    RandomCropLayer(const size_t inputWidth, const size_t inputHeight,
        const size_t channels,
        const size_t croppedWidth, const size_t croppedHeight,
        const size_t randSeed = Rand::getDefaultSeed()) :

        Layer<T>(inputWidth * inputHeight * channels,
            croppedWidth * croppedHeight * channels),
        mInputWidth(inputWidth),
        mInputHeight(inputHeight),
        mChannels(channels),
        mOutputWidth(croppedWidth),
        mOutputHeight(croppedHeight),
        mRand(randSeed),
        mTesting(false)
    {
        assert(mOutputWidth <= mInputWidth);
        assert(mOutputHeight <= mInputHeight);
    }

    void forwardSingle(const T* x, T* y) override
    {
        if (!mTesting)
        {
            // Choose the random crop section
            size_t cropX = mRand.nextInteger<size_t>(0, mInputWidth - mOutputWidth);
            size_t cropY = mRand.nextInteger<size_t>(0, mInputHeight - mOutputHeight);

            evalSingle(x, cropX, cropY, y);
        }

        else
        {
            // Use the center of the image for testing
            size_t cropX = (mInputWidth - mOutputWidth)   / 2;
            size_t cropY = (mInputHeight - mOutputHeight) / 2;

            evalSingle(x, cropX, cropY, y);
        }
    }

    void forwardBatch(const Matrix<T>& x, Matrix<T>& y) override
    {
        if (!mTesting)
        {
            // Each sample has a different random crop
            for (size_t i = 0; i < x.getRows(); ++i)
            {
                // Choose the random crop section
                size_t cropX = mRand.nextInteger<size_t>(0, mInputWidth - mOutputWidth);
                size_t cropY = mRand.nextInteger<size_t>(0, mInputHeight - mOutputHeight);

                evalSingle(x(i), cropX, cropY, y(i));
            }
        }

        else
        {
            // Use the center of the image for testing
            size_t cropX = (mInputWidth - mOutputWidth)   / 2;
            size_t cropY = (mInputHeight - mOutputHeight) / 2;

            for (size_t i = 0; i < x.getRows(); ++i)
            {
                evalSingle(x(i), cropX, cropY, y(i));
            }
        }
    }

private:
    void evalSingle(const T* x, const size_t cropX, const size_t cropY, T* y)
    {
        T* dest = y;

        for (size_t channel = 0; channel < mChannels; ++channel)
        {
            // Start at the given row in the dataset. Add a channel
            // offset. Then, navigate to the correct starting row and
            // column.
            const T* src = x + (channel * mInputWidth * mInputHeight) +
                (cropY * mInputWidth) + cropX;

            for (size_t r = 0; r < mOutputHeight; ++r)
            {
                vCopy(src, dest, mOutputWidth);

                src  += mInputWidth;
                dest += mOutputWidth;
            }
        }
    }

public:

    std::string getName() const override
    {
        return "Random Crop Layer";
    }

    std::string* getProperties(size_t& numElements) const override
    {
        std::string* arr = new std::string[1];

        char buffer[1024];
        snprintf(buffer, 1024, "(%zux%zux%zu) -> (%zux%zux%zu)",
            mInputWidth, mInputHeight, mChannels,
            mOutputWidth, mOutputHeight, mChannels);
        arr[0] = string(buffer);

        numElements = 1;
        return arr;
    }

    void setTesting(bool testing)
    {
        mTesting = testing;
    }

private:
    size_t mInputWidth, mInputHeight;
    size_t mChannels;
    size_t mOutputWidth, mOutputHeight;

    Rand mRand;
    bool mTesting;
};

}
#endif

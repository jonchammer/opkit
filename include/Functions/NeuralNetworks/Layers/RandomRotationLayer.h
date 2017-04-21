#ifndef RANDOM_ROTATION_LAYER_H
#define RANDOM_ROTATION_LAYER_H

#include <cmath>
#include "Layer.h"

namespace opkit
{

// This is a preprocessing layer that takes a muli-channel 2D image as input
// and produces a rotated version of the same image as ouput. A random
// rotation amount is used during training, and no rotation is used duing
// testing. This layer has no learned parameters, so it will not contribute to
// the gradient, but it should preceed all learning layers in the neural network.
template <class T>
class RandomRotationLayer : public Layer<T>
{

public:
    // Create a new Random Rotation Layer. We need to know the dimensions of the
    // input/output, as well as maximum and minimum rotation angles.
    RandomRotationLayer(const size_t width, const size_t height,
        const size_t channels,
        const double minAngleDegrees, const double maxAngleDegrees,
        const size_t seed = Rand::getDefaultSeed()) :

        Layer<T>(width * height * channels,
            width * height * channels),
        mWidth(width),
        mHeight(height),
        mChannels(channels),
        mMinAngle(minAngleDegrees * M_PI / 180.0),
        mMaxAngle(maxAngleDegrees * M_PI / 180.0),
        mRand(seed),
        mTesting(false)
    {
        // Do nothing
    }

private:

    // Rotates a single image (one row in x) by the given angle in radians.
    void evalSingle(const T* x, const double angle, T* y)
    {
        const T* src = x;
        T* dest      = y;

        double cosAngle = std::cos(-angle);
        double sinAngle = std::sin(-angle);

        for (size_t channel = 0; channel < mChannels; ++channel)
        {
            for (size_t y = 0; y < mHeight; ++y)
            {
                double ySinAngle = y * sinAngle;
                double yCosAngle = y * cosAngle;

                for (size_t x = 0; x < mWidth; ++x)
                {
                    // Rotate (x,y) by -angle
                    double xp = x * cosAngle - ySinAngle;
                    double yp = x * sinAngle + yCosAngle;

                    // Interpolate via nearest neighbor
                    int srcX  = std::round(xp);
                    int srcY  = std::round(yp);

                    // Copy the appropriate pixel
                    if (srcX < 0 || srcX >= mWidth || srcY < 0 || srcY >= mHeight)
                        *dest++ = T{};
                    else *dest++ = src[srcY * mWidth + srcX];
                }
            }

            // Advance to the next chanel
            src += mWidth * mHeight;
        }
    }

public:

    void forwardSingle(const T* x, T* y)
    {
        if (!mTesting)
        {
            // Choose the angle randomly
            double angle = mRand.nextReal(mMinAngle, mMaxAngle);
            evalSingle(x, angle, y);
        }

        else
        {
            // Use no rotation for testing. Just pass the data through unaltered.
            vCopy(x, y, mWidth * mHeight);
        }
    }

    void forwardBatch(const Matrix<T>& x, Matrix<T>& y)
    {
        if (!mTesting)
        {
            // Each sample has a different rotation
            for (size_t i = 0; i < x.getRows(); ++i)
            {
                // Choose the angle randomly
                double angle = mRand.nextReal(mMinAngle, mMaxAngle);
                evalSingle(x(i), angle, y(i));
            }
        }

        else
        {
            // Use no rotation for testing. Just pass the data through unaltered.
            vCopy(x.data(), y.data(), x.getRows() * x.getCols());
        }
    }

    std::string getName() const override
    {
        return "Random Rotation Layer";
    }

    std::string* getProperties(size_t& numElements) const override
    {
        const size_t NUM_ELEMENTS = 2;
        std::string* arr = new std::string[NUM_ELEMENTS];

        char buffer[1024];
        snprintf(buffer, 1024, "(%zux%zux%zu)",
            mWidth, mHeight, mChannels);
        arr[0] = string(buffer);

        snprintf(buffer, 1024, "R: [%.2f, %.2f]", mMinAngle, mMaxAngle);
        arr[1] = string(buffer);

        numElements = NUM_ELEMENTS;
        return arr;
    }

    void setTesting(bool testing)
    {
        mTesting = testing;
    }

    // Simple getters
    size_t getWidth()    const { return mWidth;                   }
    size_t getHeight()   const { return mHeight;                  }
    size_t getChannels() const { return mChannels;                }
    double getMinAngle() const { return mMinAngle * 180.0 / M_PI; }
    double getMaxAngle() const { return mMaxAngle * 180.0 / M_PI; }

private:
    size_t mWidth, mHeight, mChannels;
    double mMinAngle, mMaxAngle;

    Rand mRand;
    bool mTesting;
};

}
#endif

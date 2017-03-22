#ifndef RANDOM_ROTATION_LAYER_H
#define RANDOM_ROTATION_LAYER_H

#include <cassert>
#include <cmath>
#include "Layer.h"

namespace opkit
{

// This is a preprocessing layer that takes a muli-channel 2D image as input
// and produces a rotated version of the same image as ouput. A random
// rotation amount is used during training, and no rotation is used duing
// testing. This layer has no learned parameters, so it will not contribute to
// the gradient, but it should preceed all learning layers in the neural network.
// template <class T>
// class RandomRotationLayer : public Layer<T>
// {
// public:
//
//     // Allows us to use the members in the base class without specifying
//     // their complete names
//     using Layer<T>::mActivation;
//     using Layer<T>::mBatchSize;
//
//     // Create a new Random Rotation Layer. We need to know the dimensions of the
//     // input/output, as well as maximum and minimum rotation angles.
//     RandomRotationLayer(const size_t width, const size_t height,
//         const size_t channels,
//         const double minAngleDegrees, const double maxAngleDegrees,
//         const size_t batchSize, const size_t seed = Rand::getDefaultSeed()) :
//
//         Layer<T>(width * height * channels,
//             width * height * channels, batchSize),
//         mWidth(width),
//         mHeight(height),
//         mChannels(channels),
//         mMinAngle(minAngleDegrees * M_PI / 180.0),
//         mMaxAngle(maxAngleDegrees * M_PI / 180.0),
//         mRand(seed),
//         mTesting(false)
//     {
//         // Do nothing
//     }
//
// private:
//
//     // Rotates a single image (one row in x) by the given angle in radians.
//     void evalSingle(const Matrix<T>& x, const size_t row, const double angle)
//     {
//         const T* src = x(row);
//         T* dest      = mActivation(row);
//
//         double cosAngle = std::cos(-angle);
//         double sinAngle = std::sin(-angle);
//
//         for (size_t channel = 0; channel < mChannels; ++channel)
//         {
//             for (size_t y = 0; y < mHeight; ++y)
//             {
//                 double ySinAngle = y * sinAngle;
//                 double yCosAngle = y * cosAngle;
//
//                 for (size_t x = 0; x < mWidth; ++x)
//                 {
//                     // Rotate (x,y) by -angle
//                     double xp = x * cosAngle - ySinAngle;
//                     double yp = x * sinAngle + yCosAngle;
//
//                     // Interpolate via nearest neighbor
//                     int srcX  = std::round(xp);
//                     int srcY  = std::round(yp);
//
//                     // Copy the appropriate pixel
//                     if (srcX < 0 || srcX >= mWidth || srcY < 0 || srcY >= mHeight)
//                         *dest++ = T{};
//                     else *dest++ = src[srcY * mWidth + srcX];
//                 }
//             }
//
//             // Advance to the next chanel
//             src += mWidth * mHeight;
//         }
//     }
//
// public:
//     void eval(const Matrix<T>& x) override
//     {
//         if (!mTesting)
//         {
//             // Each sample has a different rotation
//             for (size_t i = 0; i < mBatchSize; ++i)
//             {
//                 // Choose the angle randomly
//                 double angle = mRand.nextReal(mMinAngle, mMaxAngle);
//                 evalSingle(x, i, angle);
//             }
//         }
//
//         else
//         {
//             // Use no rotation for testing. Just pass the data through unaltered.
//             vCopy(x.data(), mActivation.data(), mBatchSize * x.getCols());
//         }
//     }
//
//     void calculateDeltas(const Matrix<T>& x, T* destination) override
//     {
//         // TODO: Implement.
//     }
//
//     void calculateGradient(const Matrix<T>& x, T* gradient) override
//     {
//         // Do nothing. There is no gradient to calculate since there are
//         // no optimizable parameters.
//     }
//
//     size_t getNumParameters() const override
//     {
//         return 0;
//     }
//
//     std::string getName() const override
//     {
//         return "Random Rotation Layer";
//     }
//
//     // std::string getMiscString() const override
//     // {
//     //     char buffer[1024];
//     //     snprintf(buffer, 1024, "Shape: (%zux%zux%zu), Angle Range: [%.2f, %.2f]",
//     //         mWidth, mHeight, mChannels, getMinAngle(), getMaxAngle());
//     //
//     //     return string(buffer);
//     // }
//
//     void setTesting(bool testing)
//     {
//         mTesting = testing;
//     }
//
//     // Simple getters
//     size_t getWidth()    const { return mWidth;                   }
//     size_t getHeight()   const { return mHeight;                  }
//     size_t getChannels() const { return mChannels;                }
//     double getMinAngle() const { return mMinAngle * 180.0 / M_PI; }
//     double getMaxAngle() const { return mMaxAngle * 180.0 / M_PI; }
//
// private:
//     size_t mWidth, mHeight, mChannels;
//     double mMinAngle, mMaxAngle;
//
//     Rand mRand;
//     bool mTesting;
// };

}
#endif

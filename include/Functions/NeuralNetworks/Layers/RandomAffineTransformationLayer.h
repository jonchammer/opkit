#ifndef RANDOM_AFFINE_TRANSFORMATION_LAYER_H
#define RANDOM_AFFINE_TRANSFORMATION_LAYER_H

#include "Layer.h"
#include "Matrix.h"

namespace opkit
{

template <class T>
class RandomAffineTransformationLayer : public Layer<T>
{
public:

    // Allows us to use the members in the base class without specifying
    // their complete names
    using Layer<T>::mActivation;
    using Layer<T>::mBatchSize;

    RandomAffineTransformationLayer(const size_t inputWidth,
        const size_t inputHeight, const size_t channels,
        const size_t outputWidth, const size_t outputHeight,
        const size_t batchSize, const size_t randSeed = Rand::getDefaultSeed()) :

        // Initialize standard parameters
        Layer<T>(inputWidth * inputHeight * channels,
            outputWidth * outputHeight * channels, batchSize),
        mInputWidth(inputWidth), mInputHeight(inputHeight), mChannels(channels),
        mOutputWidth(outputWidth), mOutputHeight(outputHeight), mRand(randSeed),
        mTesting(false),

        // Initialize the transformation parameters
        mMinRotation(0.0), mMaxRotation(0.0),
        mMinTranslationX(0.0), mMaxTranslationX(0.0),
        mMinTranslationY(0.0), mMaxTranslationY(0.0),
        mMinScaleX(1.0), mMaxScaleX(1.0),
        mMinScaleY(1.0), mMaxScaleY(1.0),
        mMinShearX(0.0), mMaxShearX(0.0),
        mMinShearY(0.0), mMaxShearY(0.0)
    {
    }

private:
    void evalSingle(const Matrix<T>& in, const size_t row, const Matrix<T>& affineInv/*, const T tx, const T ty*/)
    {
        // Cache the matrix elements we use
        // const T a = affineInv(0, 0);
        // const T b = affineInv(0, 1);
        // const T c = affineInv(1, 0);
        // const T d = affineInv(1, 1);

        const T* src = in(row);
        T* dest      = mActivation(row);

        for (size_t channel = 0; channel < mChannels; ++channel)
        {
            for (size_t y = 0; y < mOutputHeight; ++y)
            {
                for (size_t x = 0; x < mOutputWidth; ++x)
                {
                    // // Multiply affineInv * (x, y) and interpolate with nearest neighbor
                    int srcX = (int)(affineInv(0, 0) * x + affineInv(0, 1) * y + affineInv(0, 2) + T{0.5});
                    int srcY = (int)(affineInv(1, 0) * x + affineInv(1, 1) * y + affineInv(1, 2) + T{0.5});


                    if (srcX >= 0 && srcX < mInputWidth && srcY >= 0 && srcY < mInputHeight)
                        dest[y * mOutputWidth + x] = src[srcY * mInputWidth + srcX];
                    else dest[y * mOutputWidth + x] = T{};
                }
            }

            // Move to the next channel
            src  += mInputWidth  * mInputHeight;
            dest += mOutputWidth * mOutputHeight;
        }
    }

    void generateAffineMatrix(Matrix<T>& m)
    {
        T theta    = mRand.nextReal(mMinRotation, mMaxRotation);
        T sinTheta = std::sin(theta);
        T cosTheta = std::cos(theta);
        T scaleX   = mRand.nextReal(mMinScaleX, mMaxScaleX);
        T scaleY   = mRand.nextReal(mMinScaleY, mMaxScaleY);
        T shearX   = mRand.nextReal(mMinShearX, mMaxShearX);
        T shearY   = mRand.nextReal(mMinShearY, mMaxShearY);
        T tx       = mRand.nextReal(mMinTranslationX, mMaxTranslationX);
        T ty       = mRand.nextReal(mMinTranslationY, mMaxTranslationY);
        T w2       = mInputWidth / 2.0;
        T h2       = mInputHeight / 2.0;

        // Correct order:
        // (scale * shear) * (translate) *
        // (translate_center * rotate * translate_from_center)

        // Scale + Shear
        // m(0, 0) = scaleX;
        // m(0, 1) = scaleX * shearX;
        // m(0, 2) = 0;
        //
        // m(1, 0) = scaleY * shearY;
        // m(1, 1) = scaleY;
        // m(1, 2) = 0;
        //
        // m(2, 0) = 0;
        // m(2, 1) = 0;
        // m(2, 2) = 1;

        // Translate
        // m(0, 0) = 1;
        // m(0, 1) = 0;
        // m(0, 2) = tx;
        //
        // m(1, 0) = 0;
        // m(1, 1) = 1;
        // m(1, 2) = ty;
        //
        // m(2, 0) = 0;
        // m(2, 1) = 0;
        // m(2, 2) = 1;

        // Rotate
        // m(0, 0) = cosTheta;
        // m(0, 1) = sinTheta;
        // m(0, 2) = -w2 * cosTheta + w2 - h2 * sinTheta;
        //
        // m(1, 0) = -sinTheta;
        // m(1, 1) = cosTheta;
        // m(1, 2) = -h2 * cosTheta + h2 + w2 * sinTheta;
        //
        // m(2, 0) = 0;
        // m(2, 1) = 0;
        // m(2, 2) = 1;

        // Translate + Rotate
        // m(0, 0) = cosTheta;
        // m(0, 1) = sinTheta;
        // m(0, 2) = (-w2 * cosTheta + w2 - h2 * sinTheta) + tx;
        //
        // m(1, 0) = -sinTheta;
        // m(1, 1) = cosTheta;
        // m(1, 2) = (-h2 * cosTheta + h2 + w2 * sinTheta) + ty;
        //
        // m(2, 0) = 0;
        // m(2, 1) = 0;
        // m(2, 2) = 1;

        // All
        T A = -w2 * cosTheta + w2 - h2 * sinTheta + tx;
        T B = -h2 * cosTheta + h2 + w2 * sinTheta + ty;

        m(0, 0) = scaleX * cosTheta - scaleX * shearX * sinTheta;
        m(0, 1) = scaleX * sinTheta + scaleX * shearX * cosTheta;
        m(0, 2) = scaleX * A + scaleX * shearX * B;

        m(1, 0) = scaleY * shearY * cosTheta - scaleY * sinTheta;
        m(1, 1) = scaleY * shearY * sinTheta + scaleY * cosTheta;
        m(1, 2) = scaleY * shearY * A + scaleY * B;

        m(2, 0) = 0;
        m(2, 1) = 0;
        m(2, 2) = 1;
    }

    // void generateAffineMatrix(Matrix<T>& m)
    // {
    //     // Start with an identity matrix
    //     T a = T{1};
    //     T b = T{};
    //     T c = T{};
    //     T d = T{1};
    //
    //     T a2, b2, c2, d2;
    //
    //     // [a, b] * [e, f] = [ae + bg, af + bh]
    //     // [c, d] * [g, h] = [ce + dg, cf + dh]
    //
    //     // Multiply by rotation matrix
    //     T theta    = mRand.nextReal(mMinRotation, mMaxRotation);
    //     T sinTheta = std::sin(theta);
    //     T cosTheta = std::cos(theta);
    //
    //     a2 = a *  cosTheta + b * sinTheta;
    //     b2 = a * -sinTheta + b * cosTheta;
    //     c2 = c *  cosTheta + d * sinTheta;
    //     d2 = c * -sinTheta + d * cosTheta;
    //
    //     a = a2;
    //     b = b2;
    //     c = c2;
    //     d = d2;
    //
    //     // Multiply by scale matrix
    //     // T scaleX = mRand.nextReal(mMinScaleX, mMaxScaleX);
    //     // T scaleY = mRand.nextReal(mMinScaleY, mMaxScaleY);
    //     //
    //     // a2 = a * scaleX;
    //     // b2 = b * scaleY;
    //     // c2 = c * scaleX;
    //     // d2 = d * scaleY;
    //     //
    //     // a = a2;
    //     // b = b2;
    //     // c = c2;
    //     // d = d2;
    //
    //     // Multiply by the shear matrix
    //     // T shearX = mRand.nextReal(mMinShearX, mMaxShearX);
    //     // T shearY = mRand.nextReal(mMinShearY, mMaxShearY);
    //     //
    //     // a2 = a + b * shearY;
    //     // b2 = b + a * shearX;
    //     // c2 = c + d * shearY;
    //     // d2 = d + c * shearX;
    //     //
    //     // a = a2;
    //     // b = b2;
    //     // c = c2;
    //     // d = d2;
    //
    //     // Copy back to m
    //     m(0, 0) = a;
    //     m(0, 1) = b;
    //     m(1, 0) = c;
    //     m(1, 1) = d;
    // }

    void invert2x2(Matrix<T>& m)
    {
        const T a = m(0, 0);
        const T b = m(0, 1);
        const T c = m(1, 0);
        const T d = m(1, 1);

        T invDeterminant = T{1} / (a * d - b * c);
        m(0, 0) =  d * invDeterminant;
        m(0, 1) = -b * invDeterminant;
        m(1, 0) = -c * invDeterminant;
        m(1, 1) =  a * invDeterminant;
    }

    void invert3x3(Matrix<T>& m)
    {
        const T a11 = m(0, 0);
        const T a12 = m(0, 1);
        const T a13 = m(0, 2);
        const T a21 = m(1, 0);
        const T a22 = m(1, 1);
        const T a23 = m(1, 2);
        const T a31 = m(2, 0);
        const T a32 = m(2, 1);
        const T a33 = m(2, 2);

        // Calculate the local 2x2 determinants
        const T det11 = a22 * a33 - a23 * a32;
        const T det12 = a13 * a32 - a12 * a33;
        const T det13 = a12 * a23 - a13 * a22;

        const T det21 = a23 * a31 - a21 * a33;
        const T det22 = a11 * a33 - a13 * a31;
        const T det23 = a13 * a21 - a11 * a23;

        const T det31 = a21 * a32 - a22 * a31;
        const T det32 = a12 * a31 - a11 * a32;
        const T det33 = a11 * a22 - a12 * a21;

        // Calculate the determinant for the entire 3x3
        const T invDet = T{1.0} /
            (a11 * det11 - a12 * (a21 * a33 - a23 * a32) + a13 * det31);

        // Save the results in m
        m(0, 0) = invDet * det11;
        m(0, 1) = invDet * det12;
        m(0, 2) = invDet * det13;
        m(1, 0) = invDet * det21;
        m(1, 1) = invDet * det22;
        m(1, 2) = invDet * det23;
        m(2, 0) = invDet * det31;
        m(2, 1) = invDet * det32;
        m(2, 2) = invDet * det33;
    }

public:
    void eval(const Matrix<T>& x) override
    {
        static Matrix<T> affine(3, 3);

        if (!mTesting)
        {
            for (size_t i = 0; i < mBatchSize; ++i)
            {
                // Create the affine transformation matrix
                generateAffineMatrix(affine);

                // Invert the affine matrix so we can fill the output image
                // from the inside out.
                // invert2x2(affine);
                invert3x3(affine);

                // Perform the transformation
                evalSingle(x, i, affine);
            }
        }
        else
        {
            // Use the identity affine transformation matrix
            affine(0, 0) = T{1};
            affine(0, 1) = T{};
            affine(0, 2) = (mInputWidth - mOutputWidth) / T{2.0};

            affine(1, 0) = T{};
            affine(1, 1) = T{1};
            affine(1, 2) = (mInputHeight - mOutputHeight) / T{2.0};

            affine(2, 0) = T{};
            affine(2, 1) = T{};
            affine(2, 2) = T{1};
            
            for (size_t i = 0; i < mBatchSize; ++i)
            {
                // Perform the transformation
                evalSingle(x, i, affine);
            }
        }
    }

    void calculateDeltas(const Matrix<T>& x, T* destination) override
    {
        // TODO: Implement.
    }

    void calculateGradient(const Matrix<T>& x, T* gradient) override
    {
        // Do nothing. There is no gradient to calculate since there are
        // no optimizable parameters.
    }

    size_t getNumParameters() const override
    {
        return 0;
    }

    std::string getName() const override
    {
        return "Random Affine Transformation Layer";
    }

    std::string getMiscString() const override
    {
        return "";
        // char buffer[1024];
        // snprintf(buffer, 1024, "(%zux%zux%zu) -> (%zux%zux%zu)",
        //     mInputWidth, mInputHeight, mChannels,
        //     mOutputWidth, mOutputHeight, mChannels);
        //
        // return string(buffer);
    }

    void setTesting(bool testing)
    {
        mTesting = testing;
    }

    void setRotationRange(double minRotation, double maxRotation)
    {
        mMinRotation = minRotation;
        mMaxRotation = maxRotation;
    }

    void setTranslationXRange(double minTranslationX, double maxTranslationX)
    {
        mMinTranslationX = minTranslationX;
        mMaxTranslationX = maxTranslationX;
    }

    void setTranslationYRange(double minTranslationY, double maxTranslationY)
    {
        mMinTranslationY = minTranslationY;
        mMaxTranslationY = maxTranslationY;
    }

    void setScaleXRange(double minScaleX, double maxScaleX)
    {
        mMinScaleX = minScaleX;
        mMaxScaleX = maxScaleX;
    }

    void setScaleYRange(double minScaleY, double maxScaleY)
    {
        mMinScaleY = minScaleY;
        mMaxScaleY = maxScaleY;
    }

    void setShearXRange(double minShearX, double maxShearX)
    {
        mMinShearX = minShearX;
        mMaxShearX = maxShearX;
    }

    void setShearYRange(double minShearY, double maxShearY)
    {
        mMinShearY = minShearY;
        mMaxShearY = maxShearY;
    }

private:
    // Layer dimensions
    size_t mInputWidth, mInputHeight;
    size_t mChannels;
    size_t mOutputWidth, mOutputHeight;

    // Training vs. Testing mode
    Rand mRand;
    bool mTesting;

    // Transformation parameters
    double mMinRotation, mMaxRotation;
    double mMinTranslationX, mMaxTranslationX;
    double mMinTranslationY, mMaxTranslationY;
    double mMinScaleX, mMaxScaleX;
    double mMinScaleY, mMaxScaleY;
    double mMinShearX, mMaxShearX;
    double mMinShearY, mMaxShearY;
};

}

#endif

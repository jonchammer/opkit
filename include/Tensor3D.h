/*
 * File:   Tensor3D.h
 * Author: Jon C. Hammer
 *
 * Created on August 20, 2016, 10:27 AM
 */

#ifndef TENSOR3D_H
#define TENSOR3D_H

#include <vector>
using std::vector;
using std::size_t;

namespace opkit
{

// This class provides a 3D array wrapper around a normal C++ vector. It handles
// the math necessary to provide the user with a 3-index interface, plus range
// checking. This is used primarily with Convolutional Models.
template <class T>
class Tensor3D
{
public:
    // Create a new Tensor that wraps the given vector (starting with
    // 'parametersStart'). The virtual dimensions are 'width'x'height'x'depth'.
    Tensor3D(vector<T>& parameters, const size_t parametersStart,
        const int width, const int height, const int depth) :
        mParameters(parameters.data() + parametersStart),
        mWidth(width), mHeight(height), mDepth(depth)
    {
        // Do nothing
    }

    Tensor3D(T* parameters, const size_t parametersStart,
        const int width, const int height, const int depth) :
        mParameters(parameters + parametersStart),
        mWidth(width), mHeight(height), mDepth(depth)
    {
        // Do nothing
    }

    // Retrieve and modify a given cell in the virtual 3D Tensor
    T get(const int x, const int y, const int z) const
    {
        if (x < 0 || y < 0 || z < 0 ||
            x >= mWidth || y >= mHeight || z >= mDepth)
            return 0;

        else
        {
            int index = (mWidth * mHeight * z + mWidth * y + x);
            return mParameters[index];
        }
    }

    void set(const int x, const int y, const int z, const T val)
    {
        if (x >= 0 && y >= 0 && z >= 0 &&
            x < mWidth && y < mHeight && z < mDepth)
        {
            int index = (mWidth * mHeight * z + mWidth * y + x);
            mParameters[index] = val;
        }
    }

    void add(const int x, const int y, const int z, const T val)
    {
        if (x >= 0 && y >= 0 && z >= 0 &&
            x < mWidth && y < mHeight && z < mDepth)
        {
            int index = (mWidth * mHeight * z + mWidth * y + x);
            mParameters[index] += val;
        }
    }

    // Getters / Setters
    int getWidth()  const
    {
        return mWidth;
    }

    int getHeight() const
    {
        return mHeight;
    }

    int getDepth()  const
    {
        return mDepth;
    }

private:
    // The vector that is wrapped (and where to start looking)
    //vector<double>& mParameters;
    T* mParameters;

    // The dimensions of this tensor (x, y, z)
    int mWidth, mHeight, mDepth;
};

};

#endif /* TENSOR3D_H */

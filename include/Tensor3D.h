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
        const size_t width, const size_t height, const size_t depth)
        : mParameters(&parameters[0]), mParametersStart(parametersStart),
        mWidth(width), mHeight(height), mDepth(depth)
    {
        // Do nothing
    }
    
    Tensor3D(T* parameters, const size_t parametersStart,
        const size_t width, const size_t height, const size_t depth)
        : mParameters(parameters), mParametersStart(parametersStart),
        mWidth(width), mHeight(height), mDepth(depth)
    {
        // Do nothing
    }
    
    // Retrieve and modify a given cell in the virtual 3D Tensor
    T get(const int x, const int y, const int z) const
    {
        if (x < 0 || y < 0 || z < 0 || 
            x >= (int)mWidth || y >= (int)mHeight || z >= (int)mDepth) 
            return 0;
        
        else
        {
            size_t index = mParametersStart + (mWidth * mHeight * z + mWidth * y + x);
            return mParameters[index];
        }
    }
    
    void set(const int x, const int y, const int z, const T val)
    {
        if (x >= 0 && y >= 0 && z >= 0 && 
            x < (int)mWidth && y < (int)mHeight && z < (int)mDepth)
        {
            size_t index = mParametersStart + (mWidth * mHeight * z + mWidth * y + x);
            mParameters[index] = val;
        }
    }
    
    void add(const int x, const int y, const int z, const T val)
    {
        if (x >= 0 && y >= 0 && z >= 0 && 
            x < (int)mWidth && y < (int)mHeight && z < (int)mDepth)
        {
            size_t index = mParametersStart + (mWidth * mHeight * z + mWidth * y + x);
            mParameters[index] += val;
        }
    }

    // Getters / Setters
    size_t getWidth()  const
    {
        return mWidth;
    }
    
    size_t getHeight() const
    {
        return mHeight;
    }
    
    size_t getDepth()  const
    {
        return mDepth;
    }
    
private:
    // The vector that is wrapped (and where to start looking)
    //vector<double>& mParameters;
    T* mParameters;
    size_t mParametersStart;
    
    // The dimensions of this tensor (x, y, z)
    size_t mWidth, mHeight, mDepth;
};

};

#endif /* TENSOR3D_H */


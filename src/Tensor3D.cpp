#include "Tensor3D.h"
#include "ActivationFunction.h"

namespace opkit
{

Tensor3D::Tensor3D(vector<double>& parameters, const size_t parametersStart, 
    const size_t width, const size_t height, const size_t depth)
    : mParameters(&parameters[0]), mParametersStart(parametersStart),
    mWidth(width), mHeight(height), mDepth(depth)
{
    // Do nothing
}

Tensor3D::Tensor3D(double* parameters, const size_t parametersStart,
    const size_t width, const size_t height, const size_t depth)
    : mParameters(parameters), mParametersStart(parametersStart),
    mWidth(width), mHeight(height), mDepth(depth)
{
    // Do nothing
}
    
double Tensor3D::get(const int x, const int y, const int z) const
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

void Tensor3D::set(const int x, const int y, const int z, const double val)
{
    if (x >= 0 && y >= 0 && z >= 0 && 
        x < (int)mWidth && y < (int)mHeight && z < (int)mDepth)
    {
        size_t index = mParametersStart + (mWidth * mHeight * z + mWidth * y + x);
        mParameters[index] = val;
    }
}

void Tensor3D::add(const int x, const int y, const int z, const double val)
{
    if (x >= 0 && y >= 0 && z >= 0 && 
        x < (int)mWidth && y < (int)mHeight && z < (int)mDepth)
    {
        size_t index = mParametersStart + (mWidth * mHeight * z + mWidth * y + x);
        mParameters[index] += val;
    }
}

size_t Tensor3D::getWidth() const
{
    return mWidth;
}

size_t Tensor3D::getHeight() const
{
    return mHeight;
}

size_t Tensor3D::getDepth() const
{
    return mDepth;
}    

};
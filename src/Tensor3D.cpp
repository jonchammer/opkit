#include "Tensor3D.h"

Tensor3D::Tensor3D(vector<double>& parameters, const size_t parametersStart, 
    const size_t width, const size_t height, const size_t depth)
    : mParameters(parameters), mParametersStart(parametersStart),
    mWidth(width), mHeight(height), mDepth(depth)
{
    
}
    
double Tensor3D::get(const int x, const int y, const int z) const
{
    if (x < 0 || y < 0 || z < 0 || x >= (int)mWidth || y >= (int)mHeight || z >= (int)mDepth) return 0;
    
    else
    {
        size_t index = mParametersStart + (mWidth * mHeight * z + mWidth * y + x);
        return mParameters[index];
    }
}

void Tensor3D::set(const int x, const int y, const int z, const double val)
{
    if (x >= 0 && y >= 0 && z >= 0 && x < (int)mWidth && y < (int)mHeight && z < (int)mDepth)
    {
        size_t index = mParametersStart + (mWidth * mHeight * z + mWidth * y + x);
        mParameters[index] = val;
    }
}


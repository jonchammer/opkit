/* 
 * File:   Tensor3D.h
 * Author: Jon C. Hammer
 *
 * Created on August 20, 2016, 10:27 AM
 */

#ifndef TENSOR3D_H
#define TENSOR3D_H

#include <vector>
using namespace std;

class Tensor3D
{
public:
    Tensor3D(vector<double>& parameters, const size_t parametersStart, 
        const size_t width, const size_t height, const size_t depth);
    
    double get(const int w, const int h, const int d) const;
    void set(const int w, const int h, const int d, const double val);
    
private:
    vector<double>& mParameters;
    size_t mParametersStart;
    
    size_t mWidth, mHeight, mDepth;
};

#endif /* TENSOR3D_H */


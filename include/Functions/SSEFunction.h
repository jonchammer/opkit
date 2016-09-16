/* 
 * File:   SSEFunction.h
 * Author: Jon C. Hammer
 *
 * Created on August 9, 2016, 9:04 AM
 */

#ifndef SSEFUNCTION_H
#define SSEFUNCTION_H

#include <vector>
#include "ErrorFunction.h"
#include "Matrix.h"
using std::vector;

// This class is an implementation of the SSE Error function.
class SSEFunction : public ErrorFunction
{
public:
    SSEFunction(Function& baseFunction);
    
    double evaluate(const Matrix& features, const Matrix& labels);
    void calculateGradientInputs(const Matrix& features, const Matrix& labels, 
        vector<double>& gradient);
    void calculateGradientParameters(const Matrix& features, 
        const Matrix& labels, vector<double>& gradient);
    void calculateHessianInputs(const Matrix& features, const Matrix& labels,
        Matrix& hessian);
    void calculateHessianParameters(const Matrix& features, 
        const Matrix& labels, Matrix& hessian);
};

#endif /* SSEFUNCTION_H */


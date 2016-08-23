/* 
 * File:   SSEFunction.h
 * Author: Jon C. Hammer
 *
 * Created on August 9, 2016, 9:04 AM
 */

#ifndef SSEFUNCTION_H
#define SSEFUNCTION_H

#include "ErrorFunction.h"
#include "Matrix.h"

class SSEFunction : public ErrorFunction
{
public:
    SSEFunction(Function& baseFunction);
    
    double evaluate(const Matrix& features, const Matrix& labels);
    void calculateJacobianInputs(const Matrix& features, const Matrix& labels, 
        Matrix& jacobian);
    void calculateJacobianParameters(const Matrix& features, 
        const Matrix& labels, Matrix& jacobian);
    void calculateHessianInputs(const Matrix& features, const Matrix& labels,
        Matrix& hessian);
    void calculateHessianParameters(const Matrix& features, 
        const Matrix& labels, Matrix& hessian);
};

#endif /* SSEFUNCTION_H */


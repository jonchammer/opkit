/* 
 * File:   HessianFreeOptimizer.h
 * Author: Jon C. Hammer
 *
 * Created on July 24, 2016, 7:51 PM
 */

#ifndef HESSIANFREEOPTIMIZER_H
#define HESSIANFREEOPTIMIZER_H

#include <vector>
#include <cmath>
#include "Trainer.h"
#include "ErrorFunction.h"
#include "Matrix.h"
#include "PrettyPrinter.h"

using std::vector;

// http://andrew.gibiansky.com/blog/machine-learning/hessian-free-optimization/
// ...

class HessianFreeOptimizer : public Trainer
{
public:
    HessianFreeOptimizer(ErrorFunction* function) : Trainer(function) {}
    
    void iterate(const Matrix& features, const Matrix& labels);
    
private:
    
    void multiplyHessian(vector<double>& x, const vector<double>& v, 
        const Matrix& features, const Matrix& labels, vector<double>& result);
    void conjugateGradient();
};

#endif /* HESSIANFREEOPTIMIZER_H */

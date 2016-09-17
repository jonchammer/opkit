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

template <class T>
class HessianFreeOptimizer : public Trainer<T>
{
public:
    HessianFreeOptimizer(ErrorFunction<T>* function) : Trainer(function) {}
    
    void iterate(const Matrix& features, const Matrix& labels);
    
private:
    
    void multiplyHessian(vector<double>& x, const vector<double>& v, 
        const Matrix& features, const Matrix& labels, vector<double>& result);
    void conjugateGradient();
};

template <class T>
void HessianFreeOptimizer<T>::iterate(const Matrix& features, const Matrix& labels)
{
    // The step size that will be used if the calculated value is unreasonable
    // (e.g. negative)
    const double DEFAULT_STEP_SIZE = 1E-6;
    
    cout << "SSE: " << Trainer<T>::function->evaluate(features, labels) << endl;
    
    // f(x) = some arbitrary nonlinear function (R^N -> R)
    // g(x) = quadratic approximation of f
    //      = ((1/2) * x^T * A * x) + (x^T * b) + (c), where
    //    A = H 
    //    b = f.grad(x) - H * x
    //    c = f(x) - x^T * f.grad(x) + (1/2)x^T * H * x
    //
    // x           = vector of size N
    // f(x) = g(x) = scalar
    // f.grad(x)   = vector of size N
    // H           = Hessian matrix (N x N)
    // A           = N x N matrix
    // b           = 1 x N vector
    // c           = scalar
    
    vector<double>& x = Trainer<T>::function->getParameters();
    size_t N          = x.size();

    // Calculate the initial direction (the negative of the gradient)
    vector<double> gradient(N);
    vector<double> direction(N);

    //model.calculateGradient(x, gradient, gradient);
    Trainer<T>::function->calculateGradientParameters(features, labels, gradient);
    for (size_t i = 0; i < N; ++i)
        direction[i] = -gradient[i];
    
    for (size_t j = 0; j < N; ++j)
    {       
        // Calculate alpha, the ideal step size. Theoretically, alpha should
        // always be >= 0, but it's possible for that to fail if the Hessian
        // calculation is inaccurate. In such cases, we pick a small default 
        // step size and use the traditional gradient descent approach.
        //
        //           -direction * gradient
        // alpha = -------------------------------
        //         direction * Hessian * direction
        double num = 0.0;
        for (size_t i = 0; i < N; ++i)
            num += direction[i] * gradient[i];

        vector<double> Ad(N);
        multiplyHessian(x, direction, features, labels, Ad);

        double denom = 0.0;
        for (size_t i = 0; i < N; ++i)
            denom += direction[i] * Ad[i];

        // Calculate alpha and ensure it's reasonable
        double alpha = -num / denom;
        if (alpha < 0) alpha = DEFAULT_STEP_SIZE;
        
        // Update the current parameter estimation by moving 'alpha' units along 
        // 'direction'
        for (size_t i = 0; i < N; ++i)
            x[i] += alpha * direction[i];

        // Calculate beta, used for updating the current direction
        //model.calculateGradient(x, gradient, gradient);
        Trainer<T>::function->calculateGradientParameters(features, labels, gradient);

        num = 0.0;
        for (size_t i = 0; i < N; ++i)
            num += gradient[i] * Ad[i];

        // Calculate beta and ensure it's reasonable. Unreasonable beta values
        // are replaced with 0, allowing the algorithm to degenerate to normal
        // gradient descent in exceptional circumstances.
        double beta = num / denom;
        beta = std::max(0.0, beta);
        
        // Update the current direction
        for (size_t i = 0; i < N; ++i)
            direction[i] = -gradient[i] + beta * direction[i];
        
        cout << "SSE: " << Trainer<T>::function->evaluate(features, labels) << endl;
    }
}

template <class T>
void HessianFreeOptimizer<T>::multiplyHessian(vector<double>& x, const vector<double>& v, 
    const Matrix& features, const Matrix& labels, vector<double>& result)
{
    const double EPSILON = 1.0E-10;
    const size_t N       = Trainer<T>::function->getNumParameters();

    // Calculate gradient 1 - grad(f(x))
    vector<double> grad1(N);
    Trainer<T>::function->getParameters().swap(x);
    //model.calculateGradient(x, result, grad1);
    Trainer<T>::function->calculateGradientParameters(features, labels, grad1);
    Trainer<T>::function->getParameters().swap(x);

    // Calculate gradient 2 - grad(f(x + epsilon * v))
    vector<double> x2(N);  
    for (size_t i = 0; i < N; ++i)
        x2[i] = x[i] + EPSILON * v[i];

    vector<double> grad2(N);
    Trainer<T>::function->getParameters().swap(x2);
    //model.calculateGradient(x, result, grad2);
    Trainer<T>::function->calculateGradientParameters(features, labels, grad2);  
    Trainer<T>::function->getParameters().swap(x2);

    // Estimate H * v using finite differences
    result.resize(N);
    for (size_t i = 0; i < N; ++i)
        result[i] = (grad2[i] - grad1[i]) / EPSILON;
    
    // Add a damping term to improve stability
//    static double lambda = 1.0;
//    
//    // calculate p
//    double p = 0.5;
//    
//    if (p < 1.0 / 4.0) lambda *= (3.0 / 2.0);
//    else if (p > 3.0 / 4.0) lambda *= (2.0 / 3.0);
//    
//    for (size_t i = 0; i < N; ++i)
//        result[i] += lambda * v[i];
}
#endif /* HESSIANFREEOPTIMIZER_H */

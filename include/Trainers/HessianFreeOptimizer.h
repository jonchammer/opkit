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
#include "Dataset.h"
#include "PrettyPrinter.h"
using std::vector;

namespace opkit
{

// http://andrew.gibiansky.com/blog/machine-learning/hessian-free-optimization/
// ...
template <class T, class Model>
class HessianFreeOptimizer : public Trainer<T, Model>
{
public:
    HessianFreeOptimizer(ErrorFunction<T, Model>* function) : Trainer<T, Model>(function) {}

    void iterate(const Dataset<T>& features, const Dataset<T>& labels);

private:

    void multiplyHessian(vector<T>& x, const vector<T>& v,
        const Dataset<T>& features, const Dataset<T>& labels, vector<T>& result);
    void conjugateGradient();
};

template <class T, class Model>
void HessianFreeOptimizer<T, Model>::iterate(const Dataset<T>& features, const Dataset<T>& labels)
{
    // The step size that will be used if the calculated value is unreasonable
    // (e.g. negative)
    const T DEFAULT_STEP_SIZE = 1E-6;

    cout << "SSE: " << Trainer<T, Model>::function->evaluate(features, labels) << endl;

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

    vector<T>& x = Trainer<T, Model>::function->getParameters();
    size_t N          = x.size();

    // Calculate the initial direction (the negative of the gradient)
    vector<T> gradient(N);
    vector<T> direction(N);

    //model.calculateGradient(x, gradient, gradient);
    Trainer<T, Model>::function->calculateGradientParameters(features, labels, gradient);
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
        T num = 0.0;
        for (size_t i = 0; i < N; ++i)
            num += direction[i] * gradient[i];

        vector<T> Ad(N);
        multiplyHessian(x, direction, features, labels, Ad);

        T denom = 0.0;
        for (size_t i = 0; i < N; ++i)
            denom += direction[i] * Ad[i];

        // Calculate alpha and ensure it's reasonable
        T alpha = -num / denom;
        if (alpha < 0) alpha = DEFAULT_STEP_SIZE;

        // Update the current parameter estimation by moving 'alpha' units along
        // 'direction'
        for (size_t i = 0; i < N; ++i)
            x[i] += alpha * direction[i];

        // Calculate beta, used for updating the current direction
        //model.calculateGradient(x, gradient, gradient);
        Trainer<T, Model>::function->calculateGradientParameters(features, labels, gradient);

        num = 0.0;
        for (size_t i = 0; i < N; ++i)
            num += gradient[i] * Ad[i];

        // Calculate beta and ensure it's reasonable. Unreasonable beta values
        // are replaced with 0, allowing the algorithm to degenerate to normal
        // gradient descent in exceptional circumstances.
        T beta = num / denom;
        beta = std::max(0.0, beta);

        // Update the current direction
        for (size_t i = 0; i < N; ++i)
            direction[i] = -gradient[i] + beta * direction[i];

        cout << "SSE: " << Trainer<T, Model>::function->evaluate(features, labels) << endl;
    }
}

template <class T, class Model>
void HessianFreeOptimizer<T, Model>::multiplyHessian(vector<T>& x, const vector<T>& v,
    const Dataset<T>& features, const Dataset<T>& labels, vector<T>& result)
{
    const T EPSILON = 1.0E-10;
    const size_t N       = Trainer<T, Model>::function->getNumParameters();

    // Calculate gradient 1 - grad(f(x))
    vector<T> grad1(N);
    Trainer<T, Model>::function->getParameters().swap(x);
    //model.calculateGradient(x, result, grad1);
    Trainer<T, Model>::function->calculateGradientParameters(features, labels, grad1);
    Trainer<T, Model>::function->getParameters().swap(x);

    // Calculate gradient 2 - grad(f(x + epsilon * v))
    vector<T> x2(N);
    for (size_t i = 0; i < N; ++i)
        x2[i] = x[i] + EPSILON * v[i];

    vector<T> grad2(N);
    Trainer<T, Model>::function->getParameters().swap(x2);
    //model.calculateGradient(x, result, grad2);
    Trainer<T, Model>::function->calculateGradientParameters(features, labels, grad2);
    Trainer<T, Model>::function->getParameters().swap(x2);

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

};

#endif /* HESSIANFREEOPTIMIZER_H */

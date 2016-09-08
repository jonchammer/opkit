#include "HessianFreeOptimizer.h"

void HessianFreeOptimizer::iterate(const Matrix& features, const Matrix& labels)
{
    // The step size that will be used if the calculated value is unreasonable
    // (e.g. negative)
    const double DEFAULT_STEP_SIZE = 1E-6;
    
    cout << "SSE: " << function->evaluate(features, labels) << endl;
    
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
    
    vector<double>& x = function->getParameters();
    size_t N          = x.size();

    // Calculate the initial direction (the negative of the gradient)
    Matrix gradient;
    gradient.setSize(1, N);
    vector<double> direction(N);

    //model.calculateGradient(x, gradient, gradient);
    function->calculateJacobianParameters(features, labels, gradient);
    for (size_t i = 0; i < N; ++i)
        direction[i] = -gradient[0][i];
    
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
            num += direction[i] * gradient[0][i];

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
        function->calculateJacobianParameters(features, labels, gradient);

        num = 0.0;
        for (size_t i = 0; i < N; ++i)
            num += gradient[0][i] * Ad[i];

        // Calculate beta and ensure it's reasonable. Unreasonable beta values
        // are replaced with 0, allowing the algorithm to degenerate to normal
        // gradient descent in exceptional circumstances.
        double beta = num / denom;
        beta = std::max(0.0, beta);
        
        // Update the current direction
        for (size_t i = 0; i < N; ++i)
            direction[i] = -gradient[0][i] + beta * direction[i];
        
        cout << "SSE: " << function->evaluate(features, labels) << endl;
    }
}

void HessianFreeOptimizer::multiplyHessian(vector<double>& x, const vector<double>& v, 
    const Matrix& features, const Matrix& labels, vector<double>& result)
{
    const double EPSILON = 1.0E-10;
    const size_t N       = function->getNumParameters();

    // Calculate gradient 1 - grad(f(x))
    Matrix grad1;
    grad1.setSize(1, N);
    function->getParameters().swap(x);
    //model.calculateGradient(x, result, grad1);
    function->calculateJacobianParameters(features, labels, grad1);
    function->getParameters().swap(x);

    // Calculate gradient 2 - grad(f(x + epsilon * v))
    vector<double> x2(N);  
    for (size_t i = 0; i < N; ++i)
        x2[i] = x[i] + EPSILON * v[i];

    Matrix grad2;
    grad2.setSize(1, N);
    function->getParameters().swap(x2);
    //model.calculateGradient(x, result, grad2);
    function->calculateJacobianParameters(features, labels, grad2);  
    function->getParameters().swap(x2);

    // Estimate H * v using finite differences
    result.resize(N);
    for (size_t i = 0; i < N; ++i)
        result[i] = (grad2[0][i] - grad1[0][i]) / EPSILON;
    
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
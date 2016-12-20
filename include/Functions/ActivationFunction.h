/*
 * File:   ActivationFunction.h
 * Author: Jon C. Hammer
 *
 * Created on August 16, 2016, 11:01 AM
 */

#ifndef ACTIVATIONFUNCTION_H
#define ACTIVATIONFUNCTION_H

#include <cmath>
#include <utility>

namespace opkit
{

// An object must possess two properties for it to be applicable as an
// activation function within a Neural Network:
// 1. It must define an 'eval' method that takes a single number as an argument.
//    This method should be statically declared and should return a single
//    number as its result.
// 2. It must define a 'deriv' method that takes two arguments. The first is the
//    parameter x. The second will always be the result of 'eval' on the same
//    argument. Some activation functions (e.g. tanh()) can avoid unnecessary
//    computations when f(x) is known. 'deriv' should also be statically
//    declared and should also return a single number as its result.
template <class T>
struct Activation
{
    virtual T eval(T) = 0;
    virtual T deriv(T, T) = 0;
};

// Traditional tanh() activation
template <class T>
struct tanhActivation : Activation<T>
{
    T eval (T x)
    {
        return tanh(x);
    }

    T deriv(T /*x*/, T fx)
    {
        return 1.0 - fx * fx;
    }
};

// See http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
// Supposedly, the scaled tanh() function better avoids saturation when the
// inputs are already in the range [-1, 1]
template <class T>
struct scaledTanhActivation
{
    static T eval (T x)
    {
        return 1.7159 * tanh(0.666666666 * x);
    }

    static T deriv(T /*x*/, T fx)
    {
        // Since fx = 1.71519 * tanh(bx), we need to divide fx by 1.7159 to get
        // tanh(bx) by itself. We also turn the division into a multiplication
        // for better performance. 1/1.7159 ~= 0.5827.
        T tanhbx = 0.58278454455 * fx;
        return 1.14393333333 * (1.0 - tanhbx * tanhbx);
    }
};

template <class T>
struct logisticActivation
{
    static T eval (T x)
    {
        return 1.0 / (1.0 + std::exp(-x));
    }

    static T deriv(T /*x*/, T fx)
    {
        return fx * (1.0 - fx);
    }
};

template <class T>
struct linearActivation
{
    static T eval (T x)
    {
        return x;
    }

    static T deriv(T /*x*/, T /*fx*/)
    {
        return 1.0;
    }
};

template <class T>
struct reluActivation
{
    static T eval (T x)
    {
        return x < 0.0 ? 0.0 : x;
    }

    static T deriv(T x, T /*fx*/)
    {
        return x < 0.0 ? 0.0 : 1.0;
    }
};

template <class T>
struct softPlusActivation
{
    static T eval (T x)
    {
        return std::log(1.0 + std::exp(x));
    }

    static T deriv(T x, T /*fx*/)
    {
        return 1.0 / (1.0 + std::exp(-x));
    }
};

template <class T>
struct bentIdentityActivation
{
    static T eval (T x)
    {
        return 0.5 * (std::sqrt(x * x + 1.0) - 1.0) + x;
    }

    static T deriv(T x, T /*fx*/)
    {
        return x / (2.0 * std::sqrt(x * x + 1.0)) + 1.0;
    }
};

template <class T>
struct sinActivation
{
    static T eval (T x)
    {
        return std::sin(x);
    }

    static T deriv(T x, T /*fx*/)
    {
        return std::cos(x);
    }
};

};

#endif /* ACTIVATIONFUNCTION_H */

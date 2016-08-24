#include "ActivationFunction.h"

// Common activation functions with their derivatives
double tanhDeriv(double /*x*/, double fx)
{
    return 1.0 - fx * fx;
}

double scaledTanh(double x)
{
    return 1.7159 * tanh(0.666666666 * x);
}

double scaledTanhDeriv(double /*x*/, double fx)
{
    // 0.58... = 1/1.7159
    double tanhbx = 0.58278454455 * fx;
    return 1.1439333333 * (1.0 - tanhbx * tanhbx);
}

double logistic(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

double logisticDeriv(double /*x*/, double fx)
{
    return fx * (1.0 - fx);
}

double linear(double x)
{
    return x;
}

double linearDeriv(double /*x*/, double /*fx*/)
{
    return 1.0;
}

double relu(double x)
{
    return x < 0.0 ? 0.0 : x;
}

double reluDeriv(double x, double /*fx*/)
{
    return x < 0.0 ? 0.0 : 1.0;
}

double softPlus(double x)
{
    return std::log(1.0 + std::exp(x));
}

double softPlusDeriv(double x, double /*fx*/)
{
    return 1.0 / (1.0 + std::exp(-x));
}

double bentIdentity(double x)
{
    return 0.5 * (std::sqrt(x * x + 1.0) - 1.0) + x;
}

double bentIdentityDeriv(double x, double /*fx*/)
{
    return x / (2.0 * std::sqrt(x * x + 1.0)) + 1.0;
}

double sinDeriv(double x, double /*fx*/)
{
    return std::cos(x);
}

// Definition of the 'extern' activations that clients will see.
Activation tanhActivation         = {&std::tanh,    &tanhDeriv};
Activation scaledTanhActivation   = {&scaledTanh,   &scaledTanhDeriv};
Activation logisticActivation     = {&logistic,     &logisticDeriv};
Activation linearActivation       = {&linear,       &linearDeriv};
Activation reluActivation         = {&relu,         &reluDeriv};
Activation softPlusActivation     = {&softPlus,     &softPlusDeriv};
Activation bentIdentityActivation = {&bentIdentity, &bentIdentityDeriv};
Activation sinActivation          = {&std::sin,     &sinDeriv};

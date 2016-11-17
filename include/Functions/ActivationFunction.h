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

namespace athena
{
    
// Typedefs for the function pointers
typedef double (*ActivationFunc)(double);
typedef double (*ActivationDeriv)(double, double);

// An activation function and its derivative are intrinsically coupled, but we
// don't want clients to be able to change one without changing the other. A
// class could be used for this purpose, but a pair is more terse. If more
// information is ever required, a class could be used instead.
typedef std::pair<ActivationFunc, ActivationDeriv> Activation;

// To make things easy for clients, some common activations are defined here.
// Clients need only reference one of these activations in order to use them.
extern Activation tanhActivation;
extern Activation scaledTanhActivation;
extern Activation logisticActivation;
extern Activation linearActivation;
extern Activation reluActivation;
extern Activation softPlusActivation;
extern Activation bentIdentityActivation;
extern Activation sinActivation;

};

#endif /* ACTIVATIONFUNCTION_H */


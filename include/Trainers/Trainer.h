/*
 * File:   Trainer.h
 * Author: Jon C. Hammer
 *
 * Created on July 9, 2016, 9:42 PM
 */

#ifndef TRAINER_H
#define TRAINER_H

#include "ErrorFunction.h"
#include "Matrix.h"

namespace opkit
{

template <class T, class Model>
class Trainer
{
public:
    /**
     * Every Trainer must have a function on which to operate.
     * @param function. The function to be optimized.
     */
    Trainer(ErrorFunction<T, Model>* function) : function(function) {};

    /**
     * Default destructor
     */
    virtual ~Trainer() {}

    /**
     * Perform one or more steps in order to optimize the function. The exact
     * semantics of this call will depend largely on the underlying
     * implementation.
     */
    virtual void iterate(const Matrix<T>& features, const Matrix<T>& labels) = 0;

    // Setters / Getters
    ErrorFunction<T, Model>* getFunction() {return function;}

protected:
    ErrorFunction<T, Model>* function; // The function we are working with
};

};

#endif /* TRAINER_H */

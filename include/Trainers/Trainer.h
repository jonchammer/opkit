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

namespace athena
{

// TODO: Add error reporting
template <class T>
class Trainer
{
public:
    /**
     * Every Trainer must have a function on which to operate.
     * @param function. The function to be optimized.
     */
    Trainer(ErrorFunction<T>* function) : function(function) {};
    
    /**
     * Default destructor
     */
    virtual ~Trainer() {}
    
    /**
     * Perform one or more steps in order to optimize the function. The exact
     * semantics of this call will depend largely on the underlying
     * implementation.
     */
    virtual void iterate(const Matrix& features, const Matrix& labels) = 0;
    
    // Setters / Getters
    ErrorFunction<T>* getFunction() {return function;}
    
protected:
    ErrorFunction<T>* function; // The function we are working with
};

};

#endif /* TRAINER_H */


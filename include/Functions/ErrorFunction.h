/* 
 * File:   ErrorFunction.h
 * Author: Jon C. Hammer
 *
 * Created on August 12, 2016, 9:58 AM
 */

#ifndef ERRORFUNCTION_H
#define ERRORFUNCTION_H

#include "Function.h"
#include "Matrix.h"

class ErrorFunction : public Function
{
public:
    ErrorFunction(Function& baseFunction);
    
    // Error functions compare the output of a base function on a given feature
    // to a known result. The interface presented in 'Function' for these
    // methods isn't really applicable to Error Functions, so these are 
    // more intuitive replacements.
    virtual double evaluate(const Matrix& features, const Matrix& labels) = 0;
    
    // Note: The default implementations of the functions that calculate 
    // derivatives with respect to the 'inputs' are quite slow and are not
    // particularly numerically stable. Those that calculate derivatives with
    // respect to the 'parameters' are much faster and more stable, but it would
    // still be a good idea for child classes to provide better implementations
    // of all of these functions if it is possible to do so.
    // --- Use the default implementations at your own risk. ---
    virtual void calculateGradientInputs(const Matrix& features, 
        const Matrix& labels, vector<double>& gradient);
    virtual void calculateGradientParameters(const Matrix& features, 
        const Matrix& labels, vector<double>& gradient);
    virtual void calculateHessianInputs(const Matrix& features,
        const Matrix& labels, Matrix& hessian);
    virtual void calculateHessianParameters(const Matrix& features, 
        const Matrix& labels, Matrix& hessian);
    
    // Returns the number of inputs to the function and the number of outputs,
    // respectively. Error functions only have 1 output.
    virtual size_t getInputs()  const;
    virtual size_t getOutputs() const;
    
    // Our 'parameters' are simply those of the base function. We forward the
    // calls wherever necessary.
    virtual vector<double>& getParameters();
    virtual const vector<double>& getParameters() const;
    virtual size_t getNumParameters() const;
    
protected:
    Function& mBaseFunction;
    
private:    
    // The prototypes for these functions don't make much sense in the context
    // of an error function, so I've hidden the originals and exposed a new
    // interface for child classes that makes more sense for Error Functions.
    void evaluate(const vector<double>& /*input*/, 
        vector<double>& /*output*/) {}
    void calculateJacobianInputs(const vector<double>& /*x*/, Matrix& /*jacobian*/) {}
    void calculateJacobianParameters(const vector<double>& /*x*/, Matrix& /*jacobian*/) {}
    void calculateHessianInputs(const vector<double>& /*x*/, 
        const size_t /*outputIndex*/, Matrix& /*hessian*/) {}
    void calculateHessianParameters(const vector<double>& /*x*/,
        const size_t /*outputIndex*/, Matrix& /*hessian*/) {}
};

#endif /* ERRORFUNCTION_H */


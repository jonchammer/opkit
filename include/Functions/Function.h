/* 
 * File:   Model.h
 * Author: Jon C. Hammer
 *
 * Created on July 9, 2016, 7:56 PM
 */

#ifndef MODEL_H
#define MODEL_H

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include "Matrix.h"
using std::cout;
using std::endl;
using std::vector;
 
class Function
{
public:
    // ------------------------- Interface Methods ------------------------- //

    // Apply this function to the given input in order to produce an output.
    // That output will be stored in 'output'.
    virtual void evaluate(const vector<double>& input, 
        vector<double>& output) = 0;
    
    // Returns the number of inputs to the function and the number of outputs,
    // respectively.
    virtual size_t getInputs()  const = 0;
    virtual size_t getOutputs() const = 0;
    
    // Functions may be parameterized (i.e. have variables in the place of some
    // constants). Parameterized functions can be thought of as generalized
    // versions of traditional concrete functions. These functions return the
    // current parameters and the number of parameters, respectively.
    // 
    // An example of a parameterized function is the linear model: y = mx + b.
    // Here, m and b are the parameters to the function.
    virtual vector<double>& getParameters()             = 0;
    virtual const vector<double>& getParameters() const = 0;
    virtual size_t getNumParameters() const             = 0;
    
    // ---------------------- Default Implementations ---------------------- // 
    
    virtual ~Function() {}
        
    // Calculates the Jacobian of this function df(x)/dx with respect to the
    // function inputs. 'x' is the point at which the Jacobian should be 
    // calculated, and the Jacobian itself is stored in 'Jacobian'.
    virtual void calculateJacobianInputs(const vector<double>& x, 
        Matrix& jacobian);
    
    // Calculates the Jacobian of this function df(x)/dx with respect to the
    // function parameters. 'x' is the point at which the Jacobian should be 
    // calculated, and the Jacobian itself is stored in 'Jacobian'.
    virtual void calculateJacobianParameters(const vector<double>& x,
        Matrix& jacobian);
    
    // Calculates the Hessian matrix of this function with respect to the
    // function inputs. 'x' is the point at which the Hessian should be
    // calculated, and the Hessian itself is stored in 'hessian'. Since an
    // arbitrary function can have many inputs and outputs, the second
    // derivative is technically a 3rd order tensor (3D Matrix). This function
    // will only calculate the Hessian with respect to a single output, indexed
    // by 'outputIndex'. For scalar functions, 'outputIndex' should be 0.
    virtual void calculateHessianInputs(const vector<double>& x,
        const size_t outputIndex, Matrix& hessian);
    
    // Calculates the Hessian matrix of this function with respect to the
    // function parameters. 'x' is the point at which the Hessian should be
    // calculated, and the Hessian itself is stored in 'hessian'. Since an
    // arbitrary function can have many inputs and outputs, the second
    // derivative is technically a 3rd order tensor (3D Matrix). This function
    // will only calculate the Hessian with respect to a single output, indexed
    // by 'outputIndex'. For scalar functions, 'outputIndex' should be 0.
    virtual void calculateHessianParameters(const vector<double>& x,
        const size_t outputIndex, Matrix& hessian);
};

// Most functions will maintain a vector of parameters. They can inherit from
// this class in order to provide a default implementation for most functions.
class StandardFunction : public Function
{
public:
    StandardFunction(size_t inputs, size_t outputs, size_t numParams) :
        mInputs(inputs), mOutputs(outputs), mParameters(numParams) {}

    // Getters
    size_t getInputs() const                    { return mInputs;            }
    size_t getOutputs() const                   { return mOutputs;           }
    vector<double>& getParameters()             { return mParameters;        }
    const vector<double>& getParameters() const { return mParameters;        }
    size_t getNumParameters() const             { return mParameters.size(); }
        
protected:
    size_t mInputs, mOutputs;
    vector<double> mParameters;
};

// Initialize the parameters with random values from a normal distribution
// of the given mean and variance
void randomizeParameters(vector<double>& parameters, 
    const double mean, const double variance);


//class Model
//{
//public:
//    // Constructors / Destructors
//    Model();
//    Model(int numParameters);
//    Model(const Model& orig);
//    virtual ~Model() {}
//    
//    // Evaluate this function using the given input. Results are written to 'output'.
//    virtual void evaluate(const vector<double>& input, vector<double>& output) = 0;
//    
//    // Calculate the gradient of the SSE function with respect to each parameter
//    // of this model. This function should be overwritten if a more efficient implementation
//    // is available, but the default should work regardless of the underlying model.
//    virtual void calculateGradient(const vector<double>& feature, 
//        const vector<double>& label, vector<double>& gradient);
//    
//    // Same as the other version of this function, with the exception that the gradient
//    // terms will be summed for each feature/label pair. This can be overwritten,
//    // but it will usually not be required, as the default implementation will
//    // usually suffice.
//    virtual void calculateGradient(const Matrix& features, const Matrix& labels, vector<double>& gradient);
//    
//    // Measure the SSE of this model against the given testing data
//    virtual double measureSSE(const Matrix& features, const Matrix&labels);
//    
//    // Initialize the parameters with random values from a normal distribution
//    // of the given mean and variance
//    void randomizeParameters(double mean, double variance);
//    
//    // Getters / Setters
//    vector<double>& getParameters()             { return mParameters;                }
//    const vector<double>& getParameters() const { return mParameters;                }
//    int getNumParameters()                      { return mParameters.size();         }
//    void setNumParameters(int numParameters)    { mParameters.resize(numParameters); }
//    
//protected: 
//    
//    // Each model contains some number of parameters that need to be optimized.
//    // For example, a linear model would need a slope and a bias (m, b).
//    vector<double> mParameters;
//};

#endif /* MODEL_H */


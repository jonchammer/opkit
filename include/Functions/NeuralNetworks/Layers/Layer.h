/*
 * File:   Layer.h
 * Author: Jon C. Hammer
 *
 * Created on September 8, 2016, 5:26 PM
 */

#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <iomanip>
#include "Matrix.h"
#include "Acceleration.h"

namespace opkit
{

// This is the base class from which all Layers derive. The template argument
// T specifies the type used for mathematical operations (e.g. float or double).
template <class T>
class Layer
{
public:
    Layer(const size_t inputs, const size_t outputs, const size_t batchSize) :
        mParameters(nullptr),
        mInputs(inputs), mOutputs(outputs), mBatchSize(batchSize),
        mActivation(batchSize, outputs), mDeltas(batchSize, outputs) {}

    // When eval is called with two arguments, we evaluate like normal, but also
    // copy the result into y
    void eval(const Matrix<T>& x, Matrix<T>& y)
    {
        eval(x);
        vCopy(mActivation.data(), y.data(), mBatchSize * mOutputs);
    }

    // 1. Feedforward step
    // When eval is called with just one argument, it is assumed that the
    // result will be placed in mActivation.
    virtual void eval(const Matrix<T>& x) = 0;

    // 2. Calculate blame terms for each node in the previous layer
    virtual void calculateDeltas(const Matrix<T>& x, T* destination) = 0;

    // 3. Calculate the gradient with respect to the parameters
    virtual void calculateGradient(const Matrix<T>& x, T* gradient) = 0;

    // Returns the number of optimizable parameters this layer uses. Some layers
    // only transform their inputs and so have 0 parameters.
    virtual size_t getNumParameters() const = 0;

    virtual string getName() const
    {
        return "Layer";
    }
    virtual string getMiscString() const
    {
        return "";
    }

    // When layers are added to the network, they are assigned a segment of the
    // network's parameters to work with. This function tells the layer which
    // segment to use for parameter storage. The layer does not own this storage.
    // It is 'leased' from the parent network.
    void assignStorage(T* parameters)
    {
        mParameters = parameters;
        onStorageAssigned();
    }

    // This method is essentially a callback that implementing Layers can use
    // if they need to know when the Layer has been assigned to a network (and
    // thus given storage parameters to use).
    virtual void onStorageAssigned() {}

    // The batch size provided in the constructor determines the maximum number
    // of rows that can be examined. By calling this function, a smaller upper
    // bound can be used (e.g. to temporarily work one sample at a time).
    void setEffectiveBatchSize(const size_t batchSize)
    {
        mBatchSize = std::min(batchSize, mDeltas.getRows());
    }

    // General layer properties
    size_t getInputs() const    { return mInputs;     }
    size_t getOutputs() const   { return mOutputs;    }
    size_t getBatchSize() const { return mBatchSize;  }
    Matrix<T>& getActivation()  { return mActivation; }
    Matrix<T>& getDeltas()      { return mDeltas;     }

protected:
    T* mParameters;           // Storage used for the parameters for this layer
    size_t mInputs, mOutputs; // The dimensions of this layer (inputs -> outputs)
    size_t mBatchSize;        // The batch size for this layer
    Matrix<T> mActivation;    // The output of this layer
    Matrix<T> mDeltas;        // The derivative of the network with respect to
                              // each output neuron.
};

};

#endif /* LAYER_H */

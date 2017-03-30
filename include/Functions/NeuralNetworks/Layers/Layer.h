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

    // Create a new Layer object of the given dimensions.
    Layer(const size_t inputs, const size_t outputs) :
        mParameters(nullptr),
        mInputs(inputs), mOutputs(outputs)
    {
        // Do nothing
    }

    virtual ~Layer()
    {
        // Do nothing
    }
    // -----------------------------------------------------------------------//
    // Single Element Methods
    // -----------------------------------------------------------------------//

    // Pass a single sample through the layer to produce an output. 'x' must
    // be of size 'inputs' and 'y' must be of size 'outputs'.
    //
    // The default implementation is an identity transformation.
    virtual void forwardSingle(const T* x, T* y)
    {
        size_t N = std::min(mInputs, mOutputs);
        vCopy(x, y, N);
        vScale(y + N , T{}, mOutputs - N);
    }

    // Use the backpropagation algorithm to calculate the derivative of the
    // network with respect to each of the inputs to this layer. 'x' is of size
    // 'inputs' and should hold the same value that was last used with
    // 'forwardSingle'. 'deltas' is of size 'outputs' and represents the
    // derivative of the network with respect to each of the output units of
    // this layer. 'y' is the result of the previous 'forwardSingle' step, and
    // has 'outputs' elements. The result will be placed in 'dest' and will be
    // of size 'inputs'.
    //
    // The default implementation performs an identity mapping.
    virtual void backpropInputsSingle(const T* x, const T* y,
        const T* deltas, T* dest)
    {
        size_t N = std::min(mInputs, mOutputs);
        vCopy(deltas, dest, N);
        vScale(dest + N , T{}, mInputs - N);
    }

    // Use the backpropagation algorithm to calculate the derivative of the
    // network with respect to each of the parameters of this layer. 'x' is of
    // size 'inputs' and should hold the same value that was last used with
    // 'forwardSingle'. 'deltas' is of size 'outputs' and represents the
    // derivative of the network with respect to each of the output units of
    // this layer. The result will be placed in 'dest' and will be of
    // size getNumParameters().
    //
    // The default implementation does nothing.
    virtual void backpropParametersSingle(const T* x, const T* deltas, T* dest)
    {
        // Do nothing
    }

    // -----------------------------------------------------------------------//
    // Batch Methods
    // -----------------------------------------------------------------------//

    // Pass a batch of 'N' samples through this layer to produce 'N' separate
    // outputs. 'x' must be an 'N x inputs' matrix, where each row is a single
    // training sample. 'y' must be an 'N x outputs' matrix, where each row
    // will hold one unique output vector.
    //
    // In the default implementation, N forward steps are performed. If it is
    // possible to more efficiently compute the batched result, this method
    // should be overwritten by the corresponding subclass.
    virtual void forwardBatch(const Matrix<T>& x, Matrix<T>& y)
    {
        for (size_t i = 0; i < x.getRows(); ++i)
            forwardSingle(x(i), y(i));
    }

    // Use the backpropagation algorithm to calculate the derivative of the
    // network with respect to each of the inputs to this layer using a batch of
    // 'N' samples. 'x' must be an 'N x inputs' matrix, where each row
    // is a single training sample. 'deltas' must be an 'N x outputs' matrix,
    // where each row is the gradient of the network with respect to each of the
    // output units units (for each training sample) in this layer. 'y' is also
    // an 'N x outputs' matrix. It is the result of the previous call to
    // 'forwardSingle'. The result will be placed in 'dest', which is an
    // 'N x inputs' matrix.
    //
    // In the default implementation, N calls to backpropInputsSingle()
    // are performed. If it is possible to more efficiently compute the batched
    // result, this method should be overwritten by the corresponding subclass.
    virtual void backpropInputsBatch(const Matrix<T>& x, const Matrix<T>& y,
        const Matrix<T>& deltas, Matrix<T>& dest)
    {
        for (size_t i = 0; i < x.getRows(); ++i)
            backpropInputsSingle(x(i), y(i), deltas(i), dest(i));
    }

    // Use the backpropagation algorithm to calculate the average derivative of
    // the network with respect to each of the parameters of this layer using a
    // batch of 'N' samples. 'x' must be an 'N x inputs' matrix, where each row
    // is a single training sample. 'deltas' must be an 'N x outputs' matrix,
    // where each row is the gradient of the network with respect to each of the
    // output units units (for each training sample) in this layer. The average
    // gradient will be placed in 'dest'.
    //
    // NOTE: If it is desired to save each of the 'N' individual gradients, the
    // user should call backpropParametersSingle to fill in each row. This
    // function will condense those gradients by returning only the average
    // (computed per column).
    //
    // In the default implementation, N calls to backpropParametersSingle()
    // are performed. If it is possible to more efficiently compute the batched
    // result, this method should be overwritten by the corresponding subclass.
    virtual void backpropParametersBatch(const Matrix<T>& x,
        const Matrix<T>& deltas, T* dest)
    {
        const size_t N = x.getRows();
        const size_t M = getNumParameters();
        if (M > 0)
        {
            // Compute the matrix of local gradients
            Matrix<T> localGradients(N, M);
            for (size_t i = 0; i < N; ++i)
                backpropParametersSingle(x(i), deltas(i), localGradients(i));

            // Multiply by the vector [1/N, 1/N, ...] to compute the average
            Matrix<T> ones(1, N, T{1});
            mtvMultiply(localGradients.data(), ones.data(), dest, N, M, T{1.0} / N);
        }
    }

    // -----------------------------------------------------------------------//
    // Parameter Storage Methods
    // -----------------------------------------------------------------------//

    // Returns the number of optimizable parameters this layer uses. The default
    // value is 0.
    virtual size_t getNumParameters() const
    {
        return 0;
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

    // -----------------------------------------------------------------------//
    // Layer Properties
    // -----------------------------------------------------------------------//

    // This method returns a human-readable name for the layer type.
    virtual std::string getName() const
    {
        return "Layer";
    }

    // This method will return a dynamically allocated array of strings
    // containing information specific to a particular layer, or a null pointer
    // if there is no extra information to communicate. 'numElements' will be
    // set to the length of this returned array.
    //
    // The caller is responsible for deallocating the memory used by this array.
    virtual std::string* getProperties(size_t& numElements) const
    {
        numElements = 0;
        return nullptr;
    }

    // Getters
    T* getParameters()             { return mParameters; }
    size_t getInputs() const       { return mInputs;     }
    size_t getOutputs() const      { return mOutputs;    }

protected:
    T* mParameters;           // Storage used for the parameters for this layer
    size_t mInputs, mOutputs; // The dimensions of this layer (inputs -> outputs)
};

};

#endif /* LAYER_H */

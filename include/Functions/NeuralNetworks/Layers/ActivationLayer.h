#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H

#include "Layer.h"
#include "Matrix.h"
#include "Acceleration.h"
#include "ActivationFunction.h"

namespace opkit
{

// This layer performs a simple element-wise transformation to the inputs.
// Therefore the input and output sizes are the same, and this layer has no
// optimizable parameters.
template <class T>
class ActivationLayer : public Layer<T>
{
public:

    // Allows us to use the members in the base class without specifying
    // their complete names
    using Layer<T>::mOutputs;

    // Create a new layer with the given size and transformation function.
    ActivationLayer(const size_t size, Activation<T>* activation,
        bool ownActivation = true) :
        Layer<T>(size, size),
        mActivationFunction(activation),
        mOwnActivation(ownActivation)
    {}

    ~ActivationLayer()
    {
        // If we own the pointer, we need to take charge of disposing of it.
        if (mOwnActivation)
        {
            delete mActivationFunction;
            mActivationFunction = nullptr;
        }
    }

    // Performs the element-wise transformation according to the given
    // transformation function.
    void forwardSingle(const T* x, T* y) override
    {
        for (size_t i = 0; i < mOutputs; ++y)
            y[i] = mActivationFunction->eval(x[i]);
    }

    // Batch implementation provided for better performance.
    void forwardBatch(const Matrix<T>& x, Matrix<T>& y) override
    {
        const T* xData = x.data();
        T* yData       = y.data();

        const size_t N = x.getRows() * mOutputs;
        for (size_t i = 0; i < N; ++i)
            yData[i] = mActivationFunction->eval(xData[i]);
    }

    // The deltas for the downstream (left) layer are simply the deltas from
    // this layer multiplied by the derivative of the transformation function.
    void backpropInputsSingle(const T* x, const T* y,
        const T* deltas, T* dest) override
    {
        for (size_t i = 0; i < mOutputs; ++i)
        {
            dest[i] = deltas[i] * mActivationFunction->deriv(x[i], y[i]);
        }
    }

    // Batch implementation provided for better performance.
    void backpropInputsBatch(const Matrix<T>& x, const Matrix<T>& y,
        const Matrix<T>& deltas, Matrix<T>& dest)
    {
        const T* xData      = x.data();
        const T* yData      = y.data();
        const T* deltasData = deltas.data();
        T* destData         = dest.data();

        for (size_t i = 0; i < x.getRows() * mOutputs; ++i)
        {
            destData[i] =
                deltasData[i] * mActivationFunction->deriv(xData[i], yData[i]);
        }
    }

    std::string getName() const override
    {
        return "Activation Layer";
    }

    std::string* getProperties(size_t& numElements) const override
    {
        numElements = 0;
        return nullptr;
    }

private:
    Activation<T>* mActivationFunction;
    bool mOwnActivation;
};

}

#endif

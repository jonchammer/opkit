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
    using Layer<T>::mActivation;
    using Layer<T>::mDeltas;
    using Layer<T>::mBatchSize;

    // Create a new layer with the given size and transformation function.
    ActivationLayer(const size_t size, const size_t batchSize,
        Activation<T>* activation, bool ownActivation = true) :
        Layer<T>(size, size, batchSize),
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
    void eval(const Matrix<T>& x) override
    {
        const T* xData = x.data();
        T* yData       = mActivation.data();

        const size_t N = mBatchSize * mOutputs;
        for (size_t i = 0; i < N; ++i)
            yData[i] = mActivationFunction->eval(xData[i]);
    }

    // The deltas for the downstream (left) layer are simply the deltas from
    // this layer multiplied by the derivative of the transformation function.
    void calculateDeltas(const Matrix<T>& x, T* destination) override
    {
        const T* input      = x.data();
        const T* deltas     = mDeltas.data();
        const T* activation = mActivation.data();

        const size_t N = mBatchSize * mOutputs;
        for (size_t i = 0; i < N; ++i)
            destination[i] = deltas[i] * mActivationFunction->deriv(input[i], activation[i]);
    }

    void calculateGradient(const Matrix<T>& x, T* gradient) override
    {
        // We have no parameters, so there is no gradient to calculate
        // for this layer.
    }

    size_t getNumParameters() const override
    {
        return 0;
    }

private:
    Activation<T>* mActivationFunction;
    bool mOwnActivation;
};

}

#endif

#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include "Layer.h"
#include "Matrix.h"
#include "Acceleration.h"

namespace opkit
{

// This layer implements the multivariate logistic function (aka Softmax). Given
// a vector of inputs, it produces a vector of outputs such that the sum of the
// values is equal to 1.0. This makes softmax layers good choices when we need
// to predict a probability distribution.
template <class T>
class SoftmaxLayer : public Layer<T>
{
public:

    // Allows us to use the members in the base class without specifying
    // their complete names
    using Layer<T>::mInputs;
    using Layer<T>::mOutputs;

    // Create a Softmax layer. All we need to know is the dimension.
    SoftmaxLayer(size_t size) :
        Layer<T>(size, size), mJacobian(size, size) {}

    void forwardSingle(const T* x, T* y) override
    {
        // Calculate the offset (for numerical stability)
        T offset = -x[vMaxIndex(x, mInputs)];

        // y_i = e^(x_i)
        T sum{};
        for (size_t i = 0; i < mOutputs; ++i)
        {
            y[i] = exp(x[i] + offset);
            sum += y[i];
        }

        // Normalize the entries such that they sum to 1.0
        vScale(y, T{1.0} / sum, mOutputs);
    }

    void backpropInputsSingle(const T* x, const T* y,
        const T* deltas, T* dest) override
    {
         T* jacobianData = mJacobian.data();

        // Destination = J * deltas, where J = Diag(y) - y*y^T
        // and y is the activation of this layer. We construct J manually by
        // first calculating y*y^T (using the outerProduct function), and then
        // adding the diagonal terms.
        //
        // NOTE 1: J should be a symmetric matrix, so when possible, we should
        // avoid repeated calculations.
        //
        // NOTE 2: J can also be expressed as
        // J_ij = y_i (delta_i_j - y_j), where delta_i_j == 1 if i == j and
        // delta_i_j == 0 if i != j.
        // This definition is more useful if the matrix has to be constructed
        // manually.
        //
        // NOTE 3: If the Cross-Entropy cost function is used and this is the
        // last layer of the network, the values placed in 'destination' should
        // equal y' - y, where y' is the activation of this layer and y is the
        // training label (in one hot representation). This is computationally
        // much easier to calculate and much more numerically stable, so it
        // should be used whenever possible.
        mJacobian.fill(T{});
        outerProduct(y, y, jacobianData, mOutputs, mOutputs, T{-1.0});

        size_t index = 0;
        for (size_t i = 0; i < mOutputs; ++i)
        {
            jacobianData[index] += y[i];
            index               += mOutputs + 1;
        }

        symmetricMvMultiply(jacobianData, deltas, dest, mOutputs);
    }

    std::string getName() const override
    {
        return "Softmax Layer";
    }

private:
    Matrix<T> mJacobian;
};

}

#endif

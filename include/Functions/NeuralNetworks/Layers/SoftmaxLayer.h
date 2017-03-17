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
    using Layer<T>::mDeltas;
    using Layer<T>::mActivation;
    using Layer<T>::mBatchSize;

    // Create a Softmax layer. All we need to know is the dimension.
    SoftmaxLayer(size_t size, size_t batchSize) :
        Layer<T>(size, size, batchSize) {}

    void eval(const Matrix<T>& x) override
    {
        for (size_t row = 0; row < mBatchSize; ++row)
        {
            const T* xData = x(row);
            T* yData       = mActivation(row);

            // Calculate the offset (for numerical stability)
            T offset = -xData[vMaxIndex(xData, mOutputs)];

            // y_i = e^(x_i)
            T sum{};
            for (size_t i = 0; i < mOutputs; ++i)
            {
                yData[i] = exp(xData[i] + offset);
                sum     += yData[i];
            }

            // Normalize the entries such that they sum to 1.0
            vScale(yData, 1.0 / sum, mOutputs);
        }
    }

    void calculateDeltas(const Matrix<T>& /*x*/, T* destination) override
    {
        static vector<T> jacobian(mOutputs * mOutputs);
        static Matrix<T> dest(destination, mBatchSize, mInputs);
        dest.setData(destination);

        // There might be a way to reduce this to a single computation, but this
        // function will very likely not be called very often, so it's not worth
        // the time to determine its exact form. As it stands, we'll just do the
        // same computation for each row individually to calculate the deltas.
        for (size_t row = 0; row < mBatchSize; ++row)
        {
            const T* deltas     = mDeltas(row);
            const T* activation = mActivation(row);
            T* work             = jacobian.data();

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
            std::fill(jacobian.begin(), jacobian.end(), T{});
            outerProduct(activation, activation, work, mOutputs, mOutputs, T{-1.0});

            size_t index = 0;
            for (size_t i = 0; i < mOutputs; ++i)
            {
                work[index] += activation[i];
                index       += mOutputs + 1;
            }

            symmetricMvMultiply(work, deltas, dest(row), mOutputs);
        }
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

    std::string getName() const override
    {
        return "Softmax Layer";
    }
};

}

#endif

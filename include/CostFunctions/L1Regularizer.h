#ifndef L1_REGULARIZER_H
#define L1_REGULARIZER_H

#include "CostFunction.h"
#include "Matrix.h"

namespace
{
    // Returns the sign of the value [-1, 0, or 1] without using any branches
    template <typename T> int sign(T val)
    {
        return (T(0) < val) - (val < T(0));
    }
}

namespace opkit
{

// This class implements an L1 regularizer as a cost function. L1 regularization
// tends to promote sparcity amongst the parameters. Since it will be unusual
// to optimize solely for sparcity, this will normally be paired with a more
// traditional cost function (e.g. SSE/CrossEntropy) via a CompoundCostFunction.
template <class T, class Model>
class L1Regularizer : public CostFunction<T, Model>
{
public:
    using CostFunction<T, Model>::mBaseFunction;
    constexpr static T DEFAULT_LAMBDA = 0.001;

    L1Regularizer(Model& baseFunction, const T lambda = DEFAULT_LAMBDA) :
        CostFunction<T, Model>(baseFunction), mLambda(lambda)
    {
        // Do nothing
    }

    T evaluate(const Matrix<T>& /*features*/, const Matrix<T>& /*labels*/)
    {
        T* params      = mBaseFunction.getParameters().data();
        const size_t N = mBaseFunction.getNumParameters();

        T sum{};
        for (size_t i = 0; i < N; ++i)
            sum += std::abs(params[i]);

        return mLambda * sum;
    }

    void calculateGradientInputs(const Matrix<T>& features,
        const Matrix<T>& labels, vector<T>& gradient)
    {
        // L1 regularization doesn't affect the gradient with respect to
        // the inputs
        std::fill(gradient.begin(), gradient.end(), T{});
    }

    void calculateGradientParameters(const Matrix<T>& features,
        const Matrix<T>& labels, vector<T>& gradient)
    {
        T* grad        = gradient.data();
        T* params      = mBaseFunction.getParameters().data();
        const size_t N = mBaseFunction.getNumParameters();

        for (size_t i = 0; i < N; ++i)
            grad[i] = sign(params[i]) * mLambda;
    }

    void setLambda(const T lambda) { mLambda = lambda; }
    T getLambda() const            { return mLambda;   }

private:
    T mLambda;
};

}

#endif

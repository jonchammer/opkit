#ifndef L2_REGULARIZER_H
#define L2_REGULARIZER_H

#include "CostFunction.h"
#include "Matrix.h"

namespace opkit
{

template <class T, class Model>
class L2Regularizer : public CostFunction<T, Model>
{
public:
    using CostFunction<T, Model>::mBaseFunction;
    constexpr static T DEFAULT_LAMBDA = 0.001;

    L2Regularizer(Model& baseFunction, const T lambda = DEFAULT_LAMBDA) :
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
            sum += params[i] * params[i];

        return mLambda * sum;
    }

    void calculateGradientInputs(const Matrix<T>& features,
        const Matrix<T>& labels, vector<T>& gradient)
    {
        // L2 regularization doesn't affect the gradient with respect to
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
            grad[i] = params[i] * mLambda;
    }

    void setLambda(const T lambda) { mLambda = lambda; }
    T getLambda() const            { return mLambda;   }

private:
    T mLambda;
};

}

#endif

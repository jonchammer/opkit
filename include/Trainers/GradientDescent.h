/*
 * File:   GradientDescent.h
 * Author: Jon C. Hammer
 *
 * Created on July 13, 2016, 9:12 AM
 */

#ifndef GRADIENTDESCENT_H
#define GRADIENTDESCENT_H

#include <vector>
#include "Trainer.h"
#include "ErrorFunction.h"
#include "Matrix.h"
#include "Acceleration.h"
#include "PrettyPrinter.h"

namespace opkit
{

// This class implements a traditional gradient descender. It looks at every
// sample in the given matrices in order to calculate the best possible estimate
// of the gradient of the error surface before taking a single step. Note that
// by using this class with a BatchIterator, both minibatches and stochastic
// sampling can be performed.
//
// NOTE: The simple gradient descender has no advanced features like Nesterov
// momentum. Instead, it is designed to be fast.
template <class T, class Model>
class SimpleGradientDescent : public Trainer<T, Model>
{
public:
    using Trainer<T, Model>::function;
    const T DEFAULT_LEARNING_RATE = 1E-4;

    SimpleGradientDescent(ErrorFunction<T, Model>* function) :
        Trainer<T, Model>(function),
        mLearningRate(DEFAULT_LEARNING_RATE) {}

    void iterate(const Matrix<T>& features, const Matrix<T>& labels)
    {
        T* params      = function->getParameters().data();
        const size_t N = function->getNumParameters();

        // Estimate the complete gradient
        static vector<T> gradient(N);
        function->calculateGradientParameters(features, labels, gradient);

        // Descend the gradient using accelerated vector operations
        vAdd(gradient.data(), params, N, -mLearningRate);
    }

    // Setters / Getters
    void setLearningRate(const T learningRate) { mLearningRate = learningRate; }
    T getLearningRate() const                  { return mLearningRate;         }

private:
    T mLearningRate;
};

// This class implements a traditional gradient descender. It looks at every
// sample in the given matrices in order to calculate the best possible estimate
// of the gradient of the error surface before taking a single step. Note that
// by using this class with a BatchIterator, both minibatches and stochastic
// sampling can be performed.
//
// NOTE: This implementation also allows the use of Nesterov momentum in order
// to improve the convergence rate. Values should be small (e.g. very close to
// 0.0) to be effective. When the momentum value is set to 0.0, no momentum will
// be used. This implementation will very likely be slightly slower than that
// used in SimpleGradientDescent.
template <class T, class Model>
class GradientDescent : public Trainer<T, Model>
{
public:
    using Trainer<T, Model>::function;
    const T DEFAULT_LEARNING_RATE = 1E-4;
    const T DEFAULT_MOMENTUM      = 1E-3;

    GradientDescent(ErrorFunction<T, Model>* function) :
        Trainer<T, Model>(function),
        mVelocity(function->getNumParameters(), 1.0),
        mLearningRate(DEFAULT_LEARNING_RATE),
        mMomentum(DEFAULT_MOMENTUM) {}

    void iterate(const Matrix<T>& features, const Matrix<T>& labels)
    {
        T* params      = function->getParameters().data();
        T* velocity    = mVelocity.data();
        const size_t N = mVelocity.size();

        // First step for Nesterov momentum.
        // params -= momentum * velocity
        vAdd(velocity, params, N, -mMomentum);

        // Estimate the complete gradient
        static vector<T> gradient(N);
        function->calculateGradientParameters(features, labels, gradient);

        // Descend the gradient (and apply momentum).
        // This operation is too complex to be performed using the accelerated
        // vector operations, so we just use the simple approach.
        for (size_t i = 0; i < N; ++i)
        {
            T temp      = velocity[i] * mMomentum;
            velocity[i] = temp + mLearningRate * gradient[i];
            params[i]  += (temp - velocity[i]);
        }
    }

    // Setters / Getters
    void setLearningRate(const T learningRate) { mLearningRate = learningRate; }
    T getLearningRate() const                  { return mLearningRate;         }

    void setMomentum(const T momentum)         { mMomentum = momentum; }
    T getMomentum() const                      { return mMomentum;     }

private:
    vector<T> mVelocity;
    T mLearningRate;
    T mMomentum;
};

};

#endif /* GRADIENTDESCENT_H */

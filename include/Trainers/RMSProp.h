/*
 * File:   RMSProp.h
 * Author: Jon C. Hammer
 *
 * Created on November 6, 2016, 10:29 AM
 */

#ifndef RMSPROP_H
#define RMSPROP_H

#include <vector>
#include <cmath>
#include "Trainer.h"
#include "ErrorFunction.h"
#include "Matrix.h"
#include "Acceleration.h"
using std::vector;

namespace opkit
{

// An implementation of batch RMS prop alone. Two parameters are used to adjust
// the performance:
//   1 - Learning Rate - same as for traditional gradient descent.
//   2 - Decay - Dictates size of moving window. When decay = 1.0, this
//       implementation reduces to traditional gradient descent. Value should be
//       between [0.0 and 1.0].
//
// See: http://climin.readthedocs.io/en/latest/rmsprop.html
//      http://sebastianruder.com/optimizing-gradient-descent/index.html
template <class T, class Model>
class SimpleRMSProp : public Trainer<T, Model>
{
public:
    const T DEFAULT_DECAY         = 0.90;
    const T DEFAULT_LEARNING_RATE = 1E-4;

    SimpleRMSProp(ErrorFunction<T, Model>* function) :
        Trainer<T, Model>(function),
        mRMS(function->getNumParameters(), 1.0),
        mDecay(DEFAULT_DECAY),
        mLearningRate(DEFAULT_LEARNING_RATE) {}

    void iterate(const Matrix<T>& features, const Matrix<T>& labels)
    {
        T* params      = Trainer<T, Model>::function->getParameters().data();
        T* RMS         = mRMS.data();
        const size_t N = mRMS.size();

        static vector<T> gradient(N);
        Trainer<T, Model>::function->calculateGradientParameters(features, labels, gradient);

        for (size_t i = 0; i < N; ++i)
        {
            // Logically, the RMS update is:
            // RMS[i] = (1.0 - mDecay) * gradient^2 + mDecay * RMS[i]
            // This is a reorganization of the same formula that has fewer operations.
            T gradSquare = gradient[i] * gradient[i];
            RMS[i]       = gradSquare + mDecay * (RMS[i] - gradSquare);

            // Descend the gradient
            // This operation is too complex to be performed using the accelerated
            // vector operations, so we just use the simple approach.
            params[i] -= (mLearningRate / std::sqrt(RMS[i] + 1E-8)) * gradient[i];
        }
    }

    void setDecay(const T decay)
    {
        mDecay = decay;
    }

    void setLearningRate(const T learningRate)
    {
        mLearningRate = learningRate;
    }

    T getDecay() const
    {
        return mDecay;
    }

    T getLearningRate() const
    {
        return mLearningRate;
    }

private:
    vector<T> mRMS;
    T mDecay;
    T mLearningRate;
};

// An implementation of batch RMS prop that includes Nesterov momentum. Three
// parameters are used to adjust the performance:
//   1 - Learning Rate - same as for traditional gradient descent.
//   2 - Decay - Dictates size of moving window. When decay = 1.0, this
//       implementation reduces to traditional gradient descent. Value should be
//       between [0.0 and 1.0].
//   3 - Momentum - Adjusts the convergence speed. When momentum = 0.0, this
//       implementation reduces to traditional gradient descent. Large values
//       can cause destabilization, so this value should be reasonably small.
//
// This implementation reduces to Adagrad when Decay = 0.0 and Momentum = 0.0.
// See: http://climin.readthedocs.io/en/latest/rmsprop.html
//      http://sebastianruder.com/optimizing-gradient-descent/index.html
template <class T, class Model>
class RMSProp : public Trainer<T, Model>
{
public:
    const T DEFAULT_DECAY         = 0.90;
    const T DEFAULT_LEARNING_RATE = 1E-4;
    const T DEFAULT_MOMENTUM      = 1E-3;

    RMSProp(ErrorFunction<T, Model>* function) :
        Trainer<T, Model>(function),
        mRMS(function->getNumParameters(), 1.0),
        mVelocity(function->getNumParameters(), 1.0),
        mDecay(DEFAULT_DECAY),
        mLearningRate(DEFAULT_LEARNING_RATE),
        mMomentum(DEFAULT_MOMENTUM) {}

    void iterate(const Matrix<T>& features, const Matrix<T>& labels)
    {
        T* params      = Trainer<T, Model>::function->getParameters().data();
        T* velocity    = mVelocity.data();
        T* RMS         = mRMS.data();
        const size_t N = mVelocity.size();

        // First step for Nesterov mMomentum
        // params -= momentum * velocity
        vAdd(velocity, params, N, -mMomentum);

        static vector<T> gradient(N);
        Trainer<T, Model>::function->calculateGradientParameters(features, labels, gradient);

        for (size_t i = 0; i < N; ++i)
        {
            // Logically, the RMS update is:
            // RMS[i] = (1.0 - mDecay) * gradient^2 + mDecay + RMS[i]
            // This is a reorganization of the same formula that has fewer operations.
            T gradSquare = gradient[i] * gradient[i];
            RMS[i]       = gradSquare + mDecay * (RMS[i] - gradSquare);

            // Descend the gradient (and apply momentum).
            // This operation is too complex to be performed using the accelerated
            // vector operations, so we just use the simple approach.
            T temp      = mMomentum * velocity[i];
            velocity[i] = temp + (mLearningRate / (std::sqrt(RMS[i] + 1E-8)) * gradient[i]);
            params[i]  += temp - velocity[i];
        }
    }

    void setDecay(const T decay)
    {
        mDecay = decay;
    }

    void setLearningRate(const T learningRate)
    {
        mLearningRate = learningRate;
    }

    void setMomentum(const T momentum)
    {
        mMomentum = momentum;
    }

    T getDecay() const
    {
        return mDecay;
    }

    T getLearningRate() const
    {
        return mLearningRate;
    }

    T getMomentum() const
    {
        return mMomentum;
    }

private:
    vector<T> mRMS;
    vector<T> mVelocity;
    T mDecay;
    T mLearningRate;
    T mMomentum;
};

};

#endif /* RMSPROP_H */

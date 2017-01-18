/*
 * File:   Adam.h
 * Author: Jon C. Hammer
 *
 * Created on January 18, 2017, 11:33 AM
 */

#ifndef ADAM_H
#define ADAM_H

#include <vector>
#include <cmath>
#include "Trainer.h"
#include "ErrorFunction.h"
#include "Dataset.h"
#include "Acceleration.h"
#include "PrettyPrinter.h"

namespace opkit
{

// This class implements the ADAM optimizer, as detailed in:
// https://arxiv.org/pdf/1412.6980.pdf
// It maintains a working estimate of the mean and variance of the gradient
// and uses that information to derive a separate learning rate for each
// optimizable parameter.
template <class T, class Model>
class Adam : public Trainer<T, Model>
{
public:
    using Trainer<T, Model>::function;
    constexpr static T DEFAULT_LEARNING_RATE = 1E-3;
    constexpr static T DEFAULT_BETA1         = 0.9;
    constexpr static T DEFAULT_BETA2         = 0.999;
    constexpr static T DEFAULT_EPSILON       = std::sqrt(std::numeric_limits<T>::epsilon());

    // Construct the optimizer with the default values for all hyperparameters
    Adam(ErrorFunction<T, Model>* function) :
        Trainer<T, Model>(function),

        mBiasedMeanEstimate(function->getNumParameters()),
        mBiasedVarianceEstimate(function->getNumParameters()),
        mLearningRate(DEFAULT_LEARNING_RATE),
        mBeta1(DEFAULT_BETA1),
        mBeta2(DEFAULT_BETA2),
        mBeta1p(DEFAULT_BETA1),
        mBeta2p(DEFAULT_BETA2),
        mEpsilon(DEFAULT_EPSILON) {}

    void iterate(const Dataset<T>& features, const Dataset<T>& labels)
    {
        const size_t N = function->getNumParameters();
        static vector<T> gradient(N);

        T* params = function->getParameters().data();
        T* grad   = gradient.data();
        T* mean   = mBiasedMeanEstimate.data();
        T* var    = mBiasedVarianceEstimate.data();

        // Estimate the gradient at this point with respect to the given
        // training data
        function->calculateGradientParameters(features, labels, gradient);

        for (size_t i = 0; i < N; ++i)
        {
            // Update the (biased) estimates for the mean and variance of the
            // computed gradient
            mean[i] = mBeta1 * mean[i] + (1.0 - mBeta1) * grad[i];
            var[i]  = mBeta2 * var[i]  + (1.0 - mBeta2) * grad[i] * grad[i];

            // Correct for the bias and compute a unique learning rate for this
            // parameter. Then descend the corrected gradient.
            T alpha = mLearningRate * std::sqrt(1.0 - mBeta2p) / (1.0 - mBeta1p);
            params[i] -= alpha * mean[i] / (std::sqrt(var[i]) + mEpsilon);
        }

        // BetaXp = betaX ^ p, where p is the current iteration number
        mBeta1p *= mBeta1;
        mBeta2p *= mBeta2;
    }

    // Setters / Getters
    void setLearningRate(const T learningRate) { mLearningRate = learningRate; }

    void setBeta1(const T beta1)
    {
        mBeta1  = beta1;
        mBeta1p = beta1;
    }

    void setBeta2(const T beta2)
    {
        mBeta2  = beta2;
        mBeta2p = beta2;
    }

    void setEpsilon(const T epsilon) { mEpsilon = epsilon; }

    T getLearningRate() const { return mLearningRate; }
    T getBeta1() const        { return mBeta1;        }
    T getBeta2() const        { return mBeta2;        }
    T getEpsilon() const      { return mEpsilon;      }

private:
    vector<T> mBiasedMeanEstimate;
    vector<T> mBiasedVarianceEstimate;
    T mLearningRate;
    T mBeta1, mBeta2;
    T mBeta1p, mBeta2p;
    T mEpsilon;
};

};

#endif /* ADAM_H */

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
using std::vector;

namespace opkit
{

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
        vector<T>& params = Trainer<T, Model>::function->getParameters();
        const size_t N    = params.size();
        
        for (size_t i = 0; i < N; ++i)
            params[i] -= mMomentum * mVelocity[i];
        
        static vector<T> gradient;
        Trainer<T, Model>::function->calculateGradientParameters(features, labels, gradient);
        
        for (size_t i = 0; i < N; ++i)
        {
            mRMS[i]      = (1.0 - mDecay) * gradient[i] * gradient[i] + mDecay * mRMS[i];
            T oldV       = mVelocity[i];
            mVelocity[i] = mMomentum * oldV + (mLearningRate / std::sqrt(mRMS[i] + 1E-8) * gradient[i]);
            params[i]    = params[i] + (mMomentum * oldV) - mVelocity[i];
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


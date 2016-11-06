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

const double DEFAULT_DECAY         = 0.90;
const double DEFAULT_LEARNING_RATE = 1E-4;
const double DEFAULT_MOMENTUM      = 1E-3;

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
template <class T>
class RMSProp : public Trainer<T>
{
public:
    RMSProp(ErrorFunction<T>* function) : 
        Trainer<T>(function), 
        mRMS(function->getNumParameters(), 1.0),
        mVelocity(function->getNumParameters(), 1.0),
        mDecay(DEFAULT_DECAY),
        mLearningRate(DEFAULT_LEARNING_RATE),
        mMomentum(DEFAULT_MOMENTUM) {}

    void iterate(const Matrix& features, const Matrix& labels)
    {
        vector<double>& params = Trainer<T>::function->getParameters();
        
        for (size_t i = 0; i < params.size(); ++i)
            params[i] -= mMomentum * mVelocity[i];
        
        static vector<double> gradient;
        Trainer<T>::function->calculateGradientParameters(features, labels, gradient);
        
        for (size_t i = 0; i < params.size(); ++i)
        {
            mRMS[i]      = (1.0 - mDecay) * gradient[i] * gradient[i] + mDecay * mRMS[i];
            double oldV  = mVelocity[i];
            mVelocity[i] = mMomentum * oldV + (mLearningRate / std::sqrt(mRMS[i] + 1E-8) * gradient[i]);
            params[i]    = params[i] + (mMomentum * oldV) - mVelocity[i];
        }
    }
    
    void setDecay(const double decay)
    {
        mDecay = decay;
    }
    
    void setLearningRate(const double learningRate)
    {
        mLearningRate = learningRate;
    }
    
    void setMomentum(const double momentum)
    {
        mMomentum = momentum;
    }
    
    double getDecay() const
    {
        return mDecay;
    }
    
    double getLearningRate() const
    {
        return mLearningRate;
    }
    
    double getMomentum() const
    {
        return mMomentum;
    }
    
private:
    vector<double> mRMS;
    vector<double> mVelocity;
    double mDecay;
    double mLearningRate;
    double mMomentum;
};

#endif /* RMSPROP_H */


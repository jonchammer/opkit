/* 
 * File:   StochasticGradientDescent.h
 * Author: Jon C. Hammer
 *
 * Created on July 13, 2016, 9:13 AM
 */

#ifndef STOCHASTICGRADIENTDESCENT_H
#define STOCHASTICGRADIENTDESCENT_H

#include <vector>
#include <algorithm>
#include "Trainer.h"
#include "ErrorFunction.h"
#include "Matrix.h"

// This class implements a batch stochastic gradient descender. Instead of 
// estimating the gradient completely, it only looks at N samples at a time, 
// where N is the batch size. Typically, this will converge faster than 
// traditional gradient descent, but at the expense of a noisier gradient 
// estimation. This class is well-suited for general use.
//
// Note: When the batch size is 1, the gradient is estimated from a single 
// sample. This is fastest, but also noisiest. When the batch size equals the 
// number of data points, this method is equivalent to traditional gradient 
// descent.
class StochasticGradientDescent : public Trainer
{
public:
    StochasticGradientDescent(ErrorFunction* function, int batchSize) : 
        Trainer(function), mLearningRate(0.0001), mBatchSize(batchSize), 
        mRand(0.0, 1.0) {} 
    
    // Makes one pass through the complete dataset
    void iterate(const Matrix& features, const Matrix& labels)
    {
        // Handle (Re-)initialization 
        if (mOrder.size() != features.rows())
            init(features, labels);
            
        // Shuffle the order
        for (int i = mOrder.size() - 1; i > 0; --i)
            std::swap(mOrder[i], mOrder[i * mRand(mRandGenerator)]);
        
        // Allocate space for the gradient and the sample gradient
        static vector<double> gradient(function->getNumParameters());
        static vector<double> sampleGradient;
        
        std::fill(gradient.begin(), gradient.end(), 0.0);
        sampleGradient.resize(function->getNumParameters());
        
        // Cache this for later
        vector<double>& params = function->getParameters();
        
        // Prepare the loop
        size_t index = 0;
        bool quit    = false;
        
        while (!quit)
        {
            // Estimate the gradient for N rows, where N is the batch size
            for (int i = 0; i < mBatchSize; ++i)
            {
                // Prepare this batch sample
                vector<double>& origFeature = (vector<double>&) features[mOrder[index]];
                vector<double>& origLabel   = (vector<double>&) labels[mOrder[index]];
                
                // Swap the current feature/label pair into the working matrices
                mBatchFeatures[0].swap(origFeature);
                mBatchLabels[0].swap(origLabel);
                
                // Calculate the gradient based on this one sample
                function->calculateGradientParameters(mBatchFeatures, 
                    mBatchLabels, sampleGradient);
                
                // Swap the feature/label pair back into their original
                // positions
                mBatchFeatures[0].swap(origFeature);
                mBatchLabels[0].swap(origLabel);
                
                ++index;
                
                // Add the results to the running total
                std::transform(gradient.begin(), gradient.end(), 
                    sampleGradient.begin(), gradient.begin(), std::plus<double>());

                // On the last batch, signal that we're ready to be done with this epoch
                if (index >= features.rows())
                {
                    index = 0;
                    quit  = true;
                }
            }

            // Descend the gradient
            for (size_t i = 0; i < gradient.size(); ++i)
                params[i] -= mLearningRate * gradient[i];
        }
    }
    
    // Setters / Getters
    void setLearningRate(double learningRate) { mLearningRate = learningRate; } 
    double getLearningRate()                  { return mLearningRate;         }
    
private:
    double mLearningRate;                          // The size of steps to be taken
    
    int mBatchSize;                                // The size of the batch [0, features.rows())
    Matrix mBatchFeatures;                         // Temporary storage for each batch
    Matrix mBatchLabels;                           // Temporary storage for each batch
    
    vector<int> mOrder;                            // Dictates the order in which samples are examined   
    std::default_random_engine mRandGenerator;     // Used to create a random order
    std::uniform_real_distribution<double> mRand;  // Used to create a random order
    
    void init(const Matrix& features, const Matrix& labels)
    {
        // Make sure the order vector starts out [0, 1, 2, ...]
        mOrder.resize(features.rows());
        for (size_t i = 0; i < features.rows(); ++i)
            mOrder[i] = i;

        // Make sure the temporary matrices are set up correctly
        mBatchFeatures.copyMetaData(features);
        mBatchLabels.copyMetaData(labels);
        mBatchFeatures.newRows(1);
        mBatchLabels.newRows(1);
    }
};

#endif /* STOCHASTICGRADIENTDESCENT_H */


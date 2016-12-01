/* 
 * File:   HillClimber.h
 * Author: Jon C. Hammer
 *
 * Created on July 13, 2016, 9:11 AM
 */

#ifndef HILLCLIMBER_H
#define HILLCLIMBER_H

#include <cmath>
#include <limits>
#include "Trainer.h"
#include "ErrorFunction.h"
#include "Matrix.h"

namespace opkit
{

// This class implements a hill climber. It doesn't rely on gradient information
// in order to improve the function. Instead, it tries 4 variations in each 
// dimension and chooses the one that produces the best results. This method 
// converges extremely quickly for convex functions, but is more susceptible to 
// getting stuck in local minima than some other approaches.
template <class T>
class HillClimber : public Trainer<T>
{
public:
    HillClimber(ErrorFunction<T>* function) : Trainer(function)
    {
        mStepSize.resize(function->getNumParameters());
        std::fill(mStepSize.begin(), mStepSize.end(), 0.1);
    }
    
    void iterate(const Matrix& features, const Matrix& labels)
    {
        const static double CHANGES [4] = {-1.25, -0.8, 0.8, 1.25};
        
        // Try 4 change values and pick the one that does the best
        // for each parameter
        vector<double>& params = Trainer<T>::function->getParameters();
        for (size_t i = 0; i < params.size(); ++i)
        {
            double orig     = params[i];
            int minIndex    = -1;
            double minError = Trainer<T>::function->evaluate(features, labels);

            // Try each change and record the best one.
            for (int j = 0; j < 4; ++j)
            {
                params[i]   += (CHANGES[j] * mStepSize[i]);
                double error = Trainer<T>::function->evaluate(features, labels);
                params[i]    = orig;

                if (error < minError)
                {
                    minIndex = j;
                    minError = error;
                }
            }
            
            // One of the options was better than what we already
            // had. Adjust the parameters and step size appropriately
            if (minIndex != -1)
            {
                params[i]    += (CHANGES[minIndex] * mStepSize[i]);
                mStepSize[i] *= fabs(CHANGES[minIndex]);
            }

            // The current value is best. Shrink the step size
            // a bit and continue;
            else mStepSize[i] *= 0.8;
        }
    }
    
private:
    vector<double> mStepSize; // The step size in each dimension
};

};

#endif /* HILLCLIMBER_H */


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

// This class implements a traditional gradient descender. It looks at every
// sample in order to calculate the best possible estimate of the gradient of 
// the error surface before taking a single step. This is better suited for
// noisy error surfaces, but is generally slower than Stochastic Gradient Descent.
class GradientDescent : public Trainer
{
public:
    GradientDescent(ErrorFunction* function) : Trainer(function), mLearningRate(0.0001) {}

    void iterate(const Matrix& features, const Matrix& labels)
    {
        // Estimate the complete gradient
        static vector<double> gradient;
        function->calculateGradientParameters(features, labels, gradient);

        // Descend the gradient
        vector<double>& params = function->getParameters();
        for (size_t i = 0; i < gradient.size(); ++i)
            params[i] -= mLearningRate * gradient[i];
    }

    // Setters / Getters
    void setLearningRate(double learningRate) { mLearningRate = learningRate; }
    double getLearningRate()                  { return mLearningRate;         }

private:
    double mLearningRate;
};

#endif /* GRADIENTDESCENT_H */


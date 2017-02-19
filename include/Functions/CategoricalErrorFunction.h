/*
 * File:   CategoricalErrorFunction.h
 * Author: Jon C. Hammer
 *
 * Created on September 11, 2016, 10:17 AM
 */

#ifndef CATEGORICALERRORFUNCTION_H
#define CATEGORICALERRORFUNCTION_H

#include "ErrorFunction.h"
#include "Matrix.h"
#include <iostream>
using std::cout;
using std::endl;

namespace opkit
{

template <class T, class Model>
class CategoricalErrorFunction : public ErrorFunction<T, Model>
{
public:

    using ErrorFunction<T, Model>::mBaseFunction;

    CategoricalErrorFunction(Model& baseFunction) :
        ErrorFunction<T, Model>(baseFunction)
    {
        // Do nothing
    }

    T evaluate(const Matrix<T>& features, const Matrix<T>& labels)
    {
        // Initialize variables
        int misclassifications = 0;
        static vector<T> prediction(labels.getCols(), T{});
        std::fill(prediction.begin(), prediction.end(), T{});
        prediction.resize(labels.getCols());

        // Calculate the SSE
        for (size_t i = 0; i < features.getRows(); ++i)
        {
            mBaseFunction.evaluate(features(i), prediction.data());

            // Determine the largest output in the prediction
            size_t largest = 0;
            T max          = prediction[0];

            for (size_t j = 1; j < labels.getCols(); ++j)
            {
                if (prediction[j] > max)
                {
                    largest = j;
                    max     = prediction[j];
                }
            }

            // If the max column from the prediction does not coincide with
            // the '1' in the label, we have a misclassification
            if (labels(i, largest) != 1.0)
                misclassifications++;
        }

        return misclassifications;
    }
};

};
#endif /* CATEGORICALERRORFUNCTION_H */

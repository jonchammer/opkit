/*
 * File:   CategoricalErrorFunction.h
 * Author: Jon C. Hammer
 *
 * Created on September 11, 2016, 10:17 AM
 */

#ifndef CATEGORICALERRORFUNCTION_H
#define CATEGORICALERRORFUNCTION_H

#include "ErrorFunction.h"
#include "Dataset.h"

namespace opkit
{

template <class T, class Model>
class CategoricalErrorFunction : public ErrorFunction<T, Model>
{
public:
    CategoricalErrorFunction(Model& baseFunction);
    T evaluate(const Dataset<T>& features, const Dataset<T>& labels);
};

template <class T, class Model>
CategoricalErrorFunction<T, Model>::CategoricalErrorFunction(Model& baseFunction)
    : ErrorFunction<T, Model>(baseFunction)
{
    // Do nothing
}

template <class T, class Model>
T CategoricalErrorFunction<T, Model>::evaluate(const Dataset<T>& features, const Dataset<T>& labels)
{
    // Initialize variables
    int misclassifications = 0;
    static vector<T> prediction(labels.cols(), T{});
    std::fill(prediction.begin(), prediction.end(), T{});
    prediction.resize(labels.cols());

    // Calculate the SSE
    for (size_t i = 0; i < features.rows(); ++i)
    {
        ErrorFunction<T, Model>::mBaseFunction.evaluate(features[i], prediction);

        // Determine the largest output in the prediction
        size_t largest = 0;
        T max          = prediction[0];

        for (size_t j = 1; j < labels.cols(); ++j)
        {
            if (prediction[j] > max)
            {
                largest = j;
                max     = prediction[j];
            }
        }

        // If the max column from the prediction does not coincide with
        // the '1' in the label, we have a misclassification
        if (labels[i][largest] != 1.0)
            misclassifications++;
    }

    return misclassifications;
}

};
#endif /* CATEGORICALERRORFUNCTION_H */

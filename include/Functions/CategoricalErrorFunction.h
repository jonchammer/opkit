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

template <class T>
class CategoricalErrorFunction : public ErrorFunction<T>
{
public:
    CategoricalErrorFunction(T& baseFunction);
    double evaluate(const Matrix& features, const Matrix& labels);
};

template <class T>
CategoricalErrorFunction<T>::CategoricalErrorFunction(T& baseFunction)
    : ErrorFunction<T>(baseFunction)
{
    // Do nothing
}

template <class T>
double CategoricalErrorFunction<T>::evaluate(const Matrix& features, const Matrix& labels)
{
    // Initialize variables
    int misclassifications = 0;
    static vector<double> prediction(labels.cols(), 0.0);
    std::fill(prediction.begin(), prediction.end(), 0.0);
    prediction.resize(labels.cols());
    
    // Calculate the SSE
    for (size_t i = 0; i < features.rows(); ++i)
    {
        ErrorFunction<T>::mBaseFunction.evaluate(features[i], prediction);
                
        // Determine the largest output in the prediction
        int largest = 0;
        double max  = prediction[0];
        
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

#endif /* CATEGORICALERRORFUNCTION_H */


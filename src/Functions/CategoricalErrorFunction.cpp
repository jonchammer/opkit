/* 
 * File:   CategoricalErrorFunction.cpp
 * Author: Jon C. Hammer
 *
 * Created on September 11, 2016, 10:18 AM
 */

#include "CategoricalErrorFunction.h"
#include "PrettyPrinter.h"

CategoricalErrorFunction::CategoricalErrorFunction(Function& baseFunction)
    : ErrorFunction(baseFunction)
{
    // Do nothing
}

double CategoricalErrorFunction::evaluate(const Matrix& features, const Matrix& labels)
{
    // Initialize variables
    int misclassifications = 0;
    static vector<double> prediction(labels.cols(), 0.0);
    prediction.resize(labels.cols());
    
    // Calculate the SSE
    for (size_t i = 0; i < features.rows(); ++i)
    {
        mBaseFunction.evaluate(features[i], prediction);
                
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
/*
 * File:   CategoricalErrorFunction.h
 * Author: Jon C. Hammer
 *
 * Created on September 11, 2016, 10:17 AM
 */

#ifndef CATEGORICALERRORFUNCTION_H
#define CATEGORICALERRORFUNCTION_H

#include "CostFunction.h"
#include "Matrix.h"
#include "NeuralNetwork.h"
#include "Acceleration.h"

#include "PrettyPrinter.h"
#include <iostream>
using std::cout;
using std::endl;

namespace opkit
{

template <class T, class Model>
class CategoricalErrorFunction : public CostFunction<T, Model>
{
public:

    using CostFunction<T, Model>::mBaseFunction;

    CategoricalErrorFunction(Model& baseFunction) :
        CostFunction<T, Model>(baseFunction) {}

    T evaluate(const Matrix<T>& features, const Matrix<T>& labels)
    {
        // Initialize variables
        int misclassifications = 0;
        static vector<T> prediction(labels.getCols(), T{});

        // Calculate the SSE
        for (size_t i = 0; i < features.getRows(); ++i)
        {
            mBaseFunction.evaluate(features(i), prediction.data());

            // Determine the largest output in the prediction
            size_t maxIndex = vMaxIndex(prediction.data(), prediction.size());

            // If the max column from the prediction does not coincide with
            // the '1' in the label, we have a misclassification
            if (labels(i, maxIndex) != 1.0)
                misclassifications++;
        }

        return misclassifications;
    }
};

// Specialization for Neural Networks. We can evaluate categorical error very
// quickly by using batched operations, which aren't possible for normal
// functions.
template <class T>
class CategoricalErrorFunction<T, NeuralNetwork<T>> : public CostFunction<T, NeuralNetwork<T>>
{
public:

    using CostFunction<T, NeuralNetwork<T>>::mBaseFunction;

    CategoricalErrorFunction(NeuralNetwork<T>& baseFunction) :
        CostFunction<T, NeuralNetwork<T>>(baseFunction) {}

    // Since Neural Networks support batch operation, we make use of it here
    // to improve runtime performance.
    T evaluate(const Matrix<T>& features, const Matrix<T>& labels)
    {
        const size_t batchSize = mBaseFunction.getLayer(0)->getDeltas().getRows();
        const size_t M         = features.getCols();
        const size_t N         = labels.getCols();

        Matrix<T> batchFeatures((T*) features.data(), batchSize, M);
        Matrix<T> batchLabels((T*) labels.data(), batchSize, N);
        static Matrix<T> predictions(batchSize, N);

        int misclassifications = 0;

        // Calculate the SSE
        size_t rows = features.getRows();
        while (rows >= batchSize)
        {
            misclassifications += evalBatch(batchFeatures, batchLabels, predictions);

            // Move to the next batch
            T* featureData = batchFeatures.data();
            T* labelData   = batchLabels.data();
            batchFeatures.setData(featureData + batchSize * M);
            batchLabels.setData(labelData + batchSize * N);

            rows -= batchSize;
        }

        // Deal with the leftover elements
        if (rows > 0)
        {
            batchFeatures.reshape(rows, M);
            batchLabels.reshape(rows, N);
            predictions.reshape(rows, N);

            misclassifications += evalBatch(batchFeatures, batchLabels, predictions);
        }

        return misclassifications;
    }

private:

    // Measures the number of misclassifications in a single batch
    int evalBatch(Matrix<T>& batchFeatures, Matrix<T>& batchLabels,
        Matrix<T>& predictions)
    {
        const size_t batchSize = batchFeatures.getRows();
        const size_t N         = batchLabels.getCols();

        // Evaluate this minibatch
        mBaseFunction.evaluateBatch(batchFeatures, predictions);

        int misclassifications = 0;
        for (size_t i = 0; i < batchSize; ++i)
        {
            size_t maxIndex = vMaxIndex(predictions(i), N);

            // If the max column from the prediction does not coincide with
            // the '1' in the label, we have a misclassification
            if (batchLabels(i, maxIndex) != 1.0)
                misclassifications++;
        }

        return misclassifications;
    }
};

};
#endif /* CATEGORICALERRORFUNCTION_H */

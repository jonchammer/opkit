/*
 * File:   Normalizer.h
 * Author: Jon C. Hammer
 *
 * Created on July 20, 2016, 4:45 PM
 */

#ifndef NORMALIZER_H
#define NORMALIZER_H

#include <vector>
#include <cmath>
#include "Dataset.h"
using std::vector;

namespace opkit
{

// Scale an individual column from the range [min, max] to the new
// range [desiredMin, desiredMax].
template <class T>
void scaleColumn(Dataset<T>& dataset, size_t column, T min, T max,
    T desiredMin, T desiredMax)
{
    T denom = max - min;
    if (denom < 1E-6) denom = 1.0;

    T multiplier = (desiredMax - desiredMin) / denom;
    T bias       = (max * desiredMin - min * desiredMax) / denom;

    for (size_t row = 0; row < dataset.rows(); ++row)
        dataset[row][column] = dataset[row][column] * multiplier + bias;
}

// Scale an individual column to the new range [desiredMin, desiredMax].
template <class T>
void scaleColumn(Dataset<T>& dataset, size_t column, T desiredMin, T desiredMax)
{
    scaleColumn(dataset, column,
        dataset.columnMin(column), dataset.columnMax(column),
        desiredMin, desiredMax);
}

// Scale all columns in the given Dataset from the range [min, max] to the new
// range [desiredMin, desiredMax].
template <class T>
void scaleAllColumns(Dataset<T>& dataset, T min, T max, T desiredMin, T desiredMax)
{
    for (size_t i = 0; i < dataset.cols(); ++i)
        scaleColumn(dataset, i, min, max, desiredMin, desiredMax);
}

// Scale all columns in the given Dataset to the range [desiredMin, desiredMax].
template <class T>
void scaleAllColumns(Dataset<T>& dataset, T desiredMin, T desiredMax)
{
    for (size_t i = 0; i < dataset.cols(); ++i)
    {
        scaleColumn(dataset, i, dataset.columnMin(i),
            dataset.columnMax(i), desiredMin, desiredMax);
    }
}

// Adjusts a given column in a Dataset such that it has a variance of 1.0.
template <class T>
void normalizeVarianceColumn(Dataset<T>& dataset, size_t column)
{
    // Calculate the variance
    T sum  = 0.0;
    T mean = dataset.columnMean(column);
    for (size_t i = 0; i < dataset.rows(); ++i)
    {
        T temp = dataset[i][column] - mean;
        sum += temp * temp;
    }

    // Divide by the standard deviation so the data now has a variance of 1.0.
    // For uniform columns, we leave the data alone.
    if (sum > 1E-6)
    {
        T invSqrt = std::sqrt((dataset.rows() - 1) / sum);
        for (size_t i = 0; i < dataset.rows(); ++i)
            dataset[i][column] *= invSqrt;
    }
}

// Adjusts all columns of the given Dataset such that they have a variance of 1.0.
template <class T>
void normalizeVarianceAllColumns(Dataset<T>& dataset)
{
    for (size_t i = 0; i < dataset.cols(); ++i)
        normalizeVarianceColumn(dataset, i);
}

// Returns a new Dataset resembling the source, with the exception that the
// given column is expanded based to a 1-hot representation. In other words,
// a categorical column will be converted into N continuous columns, where N
// is the number of categories.
template <class T>
Dataset<T> convertColumnToOneHot(const Dataset<T>& source, size_t column)
{
    int valueCount = source.valueCount(column);
    if (valueCount == 0) return source;

    Dataset<T> dest;
    dest.setSize(source.rows(), source.cols() - 1 + valueCount);
    for (size_t i = 0; i < source.rows(); ++i)
    {
        const vector<T>& sourceRow = source.row(i);
        vector<T>& destRow         = dest.row(i);

        // Copy the data before & after the affected column
        std::copy(sourceRow.begin(), sourceRow.begin() + column, destRow.begin());
        std::copy(sourceRow.begin() + column + 1, sourceRow.end(),
            destRow.begin() + column + valueCount);

        // Convert the affected column to 1 hot
        int val = (int) sourceRow[column];
        destRow[column + val] = 1.0;
    }

    return dest;
}

};

#endif /* NORMALIZER_H */

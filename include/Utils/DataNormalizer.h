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
#include "Matrix.h"
using std::vector;

// Scale an individual column from the range [min, max] to the new
// range [desiredMin, desiredMax].
void scaleColumn(Matrix& matrix, size_t column, double min, double max, 
    double desiredMin, double desiredMax)
{
    double denom = max - min;
    if (denom < 1E-6) denom = 1.0;
    
    double multiplier = (desiredMax - desiredMin) / denom;
    double bias       = (max * desiredMin - min * desiredMax) / denom;

    for (size_t row = 0; row < matrix.rows(); ++row)
        matrix[row][column] = matrix[row][column] * multiplier + bias;
}

// Scale an individual column to the new range [desiredMin, desiredMax].
void scaleColumn(Matrix& matrix, size_t column, double desiredMin, double desiredMax)
{
    scaleColumn(matrix, column, 
        matrix.columnMin(column), matrix.columnMax(column), 
        desiredMin, desiredMax);   
}

// Scale all columns in the given matrix from the range [min, max] to the new
// range [desiredMin, desiredMax].
void scaleAllColumns(Matrix& matrix, double min, double max, 
    double desiredMin, double desiredMax)
{
    for (size_t i = 0; i < matrix.cols(); ++i)
        scaleColumn(matrix, i, min, max, desiredMin, desiredMax);
}

// Scale all columns in the given matrix to the range [desiredMin, desiredMax].
void scaleAllColumns(Matrix& matrix, double desiredMin, double desiredMax)
{
    for (size_t i = 0; i < matrix.cols(); ++i)
    {
        scaleColumn(matrix, i, matrix.columnMin(i), 
            matrix.columnMax(i), desiredMin, desiredMax);
    }
}

// Adjusts a given column in a matrix such that it has a variance of 1.0.
void normalizeVarianceColumn(Matrix& matrix, size_t column)
{
    // Calculate the variance
    double sum  = 0.0;
    double mean = matrix.columnMean(column);
    for (size_t i = 0; i < matrix.rows(); ++i)
    {
        double temp = matrix[i][column] - mean;
        sum += temp * temp;
    }
    
    // Divide by the standard deviation so the data now has a variance of 1.0. 
    // For uniform columns, we leave the data alone.
    if (sum > 1E-6)
    {
        double invSqrt = std::sqrt((matrix.rows() - 1) / sum);
        for (size_t i = 0; i < matrix.rows(); ++i)
            matrix[i][column] *= invSqrt; 
    }
}

// Adjusts all columns of the given matrix such that they have a variance of 1.0.
void normalizeVarianceAllColumns(Matrix& matrix)
{
    for (size_t i = 0; i < matrix.cols(); ++i)
        normalizeVarianceColumn(matrix, i);
}

// Returns a new matrix resembling the source, with the exception that the
// given column is expanded based to a 1-hot representation. In other words,
// a categorical column will be converted into N continuous columns, where N
// is the number of categories.
Matrix convertColumnToOneHot(const Matrix& source, size_t column)
{
    int valueCount = source.valueCount(column);
    if (valueCount == 0) return source;
    
    Matrix dest;
    dest.setSize(source.rows(), source.cols() - 1 + valueCount);
    for (size_t i = 0; i < source.rows(); ++i)
    {
        const vector<double>& sourceRow = source.row(i);
        vector<double>& destRow         = dest.row(i);
        
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

#endif /* NORMALIZER_H */


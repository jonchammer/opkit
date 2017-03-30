/*
 * File:   BatchIterator.h
 * Author: Jon C. Hammer
 *
 * Created on November 9, 2016, 12:40 PM
 */

#ifndef BATCHITERATOR_H
#define BATCHITERATOR_H

#include <vector>
#include "Dataset.h"
#include "Matrix.h"
#include "Rand.h"
#include "RandomIndexIterator.h"
using std::vector;

namespace opkit
{

template <class T>
class BatchIterator
{
public:
    BatchIterator(Matrix<T>& features, Matrix<T>& labels, size_t batchSize, Rand& rand) :
        mFeatures(features), mLabels(labels),
        mBatchFeatures(batchSize, features.getCols()),
        mBatchLabels(batchSize, labels.getCols()),
        mBatchSize(batchSize),
        mIt(features.getRows()),
        mRand(rand)
    {
        reset();
    }

    bool hasNext()
    {
        return mIt.hasNext();
    }

    void next(Matrix<T>*& features, Matrix<T>*& labels)
    {
        const size_t M = mFeatures.getCols();
        const size_t N = mLabels.getCols();

        for (size_t row = 0; row < mBatchSize && mIt.hasNext(); ++row)
        {
            size_t index = mIt.next();
            mBatchFeatures.copy(mFeatures, index, 0, 1, M, row, 0);
            mBatchLabels.copy(mLabels, index, 0, 1, N, row, 0);
        }

        features = &mBatchFeatures;
        labels   = &mBatchLabels;
    }

    void reset()
    {
        mIt.reset(mRand);
    }

private:
    Matrix<T>& mFeatures, mLabels;
    Matrix<T> mBatchFeatures, mBatchLabels;
    size_t mBatchSize;

    RandomIndexIterator mIt;
    Rand& mRand;
};

};

#endif /* BATCHITERATOR_H */

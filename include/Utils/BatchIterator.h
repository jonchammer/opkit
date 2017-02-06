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
#include "Rand.h"
using std::vector;

namespace opkit
{

template <class T>
class BatchIterator
{
public:
    BatchIterator(Dataset<T>& features, Dataset<T>& labels, size_t batchSize, Rand& rand) :
        mFeatures(features),
        mLabels(labels),
        mBatchSize(batchSize),
        mRand(rand),
        mOrder(features.rows()),
        mOrderIndex(0)
    {
        // Make sure the order vector starts out [0, 1, 2, ...]
        for (size_t i = 0; i < features.rows(); ++i)
            mOrder[i] = i;

        // Make sure the temporary matrices are set up correctly
        mBatchFeatures.setSize(batchSize, features.cols());
        mBatchLabels.setSize(batchSize, labels.cols());

        reset();
    }

    bool hasNext()
    {
        return mOrderIndex < mOrder.size();
    }

    void lock(Dataset<T>*& features, Dataset<T>*& labels)
    {
        // Swap data in
        for (size_t i = mOrderIndex; (i < mOrderIndex + mBatchSize) && (i < mOrder.size()); ++i)
        {
            // Prepare this batch sample
            vector<T>& origFeature = mFeatures[mOrder[i]];
            vector<T>& origLabel   = mLabels[mOrder[i]];

            // Swap the current feature/label pair into the working matrices
            mBatchFeatures[i - mOrderIndex].swap(origFeature);
            mBatchLabels[i - mOrderIndex].swap(origLabel);
        }

        features = &mBatchFeatures;
        labels   = &mBatchLabels;
    }

    void unlock()
    {
        // Swap data back out
        for (size_t i = mOrderIndex; (i < mOrderIndex + mBatchSize) && (i < mOrder.size()); ++i)
        {
            // Prepare this batch sample
            vector<T>& origFeature = mFeatures[mOrder[i]];
            vector<T>& origLabel   = mLabels[mOrder[i]];

            // Swap the current feature/label pair into the working matrices
            mBatchFeatures[i - mOrderIndex].swap(origFeature);
            mBatchLabels[i - mOrderIndex].swap(origLabel);
        }

        // Increment indices
        mOrderIndex += mBatchSize;
    }

    void reset()
    {
        mOrderIndex = 0;

        // Shuffle the order
        for (int i = mOrder.size() - 1; i > 0; --i)
            std::swap(mOrder[i], mOrder[mRand.nextInteger(0, i)]);
    }

private:
    Dataset<T>& mFeatures, mLabels;
    Dataset<T> mBatchFeatures, mBatchLabels;
    size_t mBatchSize;
    Rand& mRand;

    vector<size_t> mOrder;
    size_t mOrderIndex;
};

};

#endif /* BATCHITERATOR_H */

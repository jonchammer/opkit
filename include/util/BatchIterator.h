/*
 * File:   BatchIterator.h
 * Author: Jon C. Hammer
 *
 * Created on November 9, 2016, 12:40 PM
 */

#ifndef BATCHITERATOR_H
#define BATCHITERATOR_H

#include <vector>
#include "acceleration/Acceleration.h"
#include "tensor/Tensor.h"
#include "util/Rand.h"
#include "util/RandomIndexIterator.h"
using std::vector;

namespace tensorlib
{

template <class T>
class BatchIterator
{
public:
    BatchIterator(Tensor<T>& features, Tensor<T>& labels, size_t batchSize, Rand& rand) :
        mFeatures(features), mLabels(labels),
        mIt(features.shape(0)),
        mRand(rand),
        mContiguous(mFeatures.contiguous() && mLabels.contiguous())
    {
        // Set up the batch tensors
        SmallVector featuresShape(features.shape());
        SmallVector labelsShape(labels.shape());
        featuresShape[0] = batchSize;
        labelsShape[0]   = batchSize;
        mBatchFeatures.resize(featuresShape.begin(), featuresShape.end());
        mBatchLabels.resize(labelsShape.begin(), labelsShape.end());
        reset();
    }

    bool hasNext()
    {
        return mIt.hasNext();
    }

    void next(Tensor<T>*& features, Tensor<T>*& labels)
    {
        const size_t batchSize = mBatchFeatures.shape(0);
        const size_t N         = mBatchFeatures.size() / batchSize;
        const size_t M         = mBatchLabels.size() / batchSize;
        T* batchFeatureRow     = mBatchFeatures.data();
        T* batchLabelRow       = mBatchLabels.data();

        // Optimize the most common case
        if (mContiguous)
        {
            for (size_t row = 0; row < batchSize && mIt.hasNext(); ++row)
            {
                size_t index = mIt.next();

                const T* featureRow = mFeatures.data() + index * N;
                const T* labelRow   = mLabels.data()   + index * M;

                vCopy(featureRow, batchFeatureRow, N);
                vCopy(labelRow,   batchLabelRow,   M);
                batchFeatureRow += N;
                batchLabelRow   += M;
            }
        }

        // But also handle non-ideal cases
        else
        {
            for (size_t row = 0; row < batchSize && mIt.hasNext(); ++row)
            {
                size_t index = mIt.next();

                Tensor<T> featureSlice = select(mFeatures, 0, index);
                Tensor<T> labelSlice   = select(mLabels,   0, index);

                std::copy(featureSlice.begin(), featureSlice.end(), batchFeatureRow);
                std::copy(labelSlice.begin(), labelSlice.end(), batchLabelRow);
                batchFeatureRow += N;
                batchLabelRow   += M;
            }
        }

        features = &mBatchFeatures;
        labels   = &mBatchLabels;
    }

    void reset()
    {
        mIt.reset(mRand);
    }

private:
    Tensor<T>& mFeatures;
    Tensor<T>& mLabels;
    Tensor<T> mBatchFeatures;
    Tensor<T> mBatchLabels;

    RandomIndexIterator mIt;
    Rand& mRand;
    bool mContiguous;
};

};

#endif /* BATCHITERATOR_H */

#ifndef CONVOLUTIONAL_2D_LAYER_H
#define CONVOLUTIONAL_2D_LAYER_H

#include "Layer.h"
#include "Matrix.h"
#include "Acceleration.h"

namespace opkit
{
    
// // Convolutional layers are often used for image processing. They take as input
// // a 3D volume of numbers and produce as output another 3D volume. Typically,
// // the three dimensions will correspond to the width of an image, the height of
// // an image, and the number of channels in the image. Convolutional layers use
// // weights that are connected to a small area, in addition to weight sharing.
// // A set of 1 or more kernels are what are learned during training.
// template <class T>
// class Convolutional2DLayer : public Layer<T>
// {
// public:
//
//     // Allows us to use the members in the base class without specifying
//     // their complete names
//     using Layer<T>::mParameters;
//     using Layer<T>::mInputs;
//     using Layer<T>::mOutputs;
//     using Layer<T>::mActivation;
//     using Layer<T>::mDeltas;
//
//     // Constructors
//
//     // Create a 2D convolutional layer with the given metaparameters.
//     //
//     // inputWidth    - The width of the input image in pixels
//     // inputHeight   - The height of the input image in pixels
//     // inputChannels - 1 for grayscale, 3 for RGB, etc.
//     // filterSize    - The size of the convolution kernel (e.g. 3 or 5)
//     // numFilters    - How many filters should be applied to the input
//     // stride        - Allows user to adjust how much the receptive fields
//     //                 overlap with one another. A value of 1 is always allowed,
//     //                 but other values can cause problems if they don't
//     //                 partition the space evenly.
//     // zeroPadding   - The input is padded with 0 or more 0's in order to adjust
//     //                 the size of the output.
//     //
//     // NOTE 1:
//     // When it is desired to have an output size that matches the input
//     // size, the padding can be calculated using the following formula:
//     //   P = (SW - S - W + F) / 2, where
//     // S = stride, W = inputWidth/inputHeight, F = filterSize
//     //
//     // NOTE 2:
//     // The input volume will be of size:
//     //   inputWidth * inputHeight * inputChannels.
//     // The output volume will be of size:
//     //   outputWidth * outputHeight * numFilters
//     // where
//     // outputWidth  = (inputWidth  - numFilters + 2 * zeroPadding) / stride + 1
//     // outputHeight = (inputHeight - numFilters + 2 * zeroPadding) / stride + 1
//     Convolutional2DLayer(
//         size_t inputWidth, size_t inputHeight, size_t inputChannels,
//         size_t filterSize, size_t numFilters, size_t stride, size_t zeroPadding) :
//
//         // Superclass constructor - input and output dimensions
//         Layer<T>(inputWidth * inputHeight * inputChannels,
//             ((inputWidth - filterSize + 2 * zeroPadding) / stride + 1) *
//             ((inputHeight - filterSize + 2 * zeroPadding) / stride + 1) *
//             numFilters),
//
//         // Paramaters
//         mInputWidth(inputWidth), mInputHeight(inputHeight),
//         mInputChannels(inputChannels), mFilterSize(filterSize),
//         mNumFilters(numFilters), mStride(stride), mZeroPadding(zeroPadding),
//         mOutputWidth((inputWidth - filterSize + 2 * zeroPadding) / stride + 1),
//         mOutputHeight((inputHeight - filterSize + 2 * zeroPadding) / stride + 1)
//     {
//         if (stride < 1)
//         {
//             cerr << "Error: Stride must be at least 1." << endl;
//             throw Ex("Unable to create convolutional layer.");
//         }
//
//         // Check to make sure the given metaparameters are actually a valid
//         // configuration. We have to make sure that this equation:
//         // (inputWidth - filterSize + (2 * zeroPadding)) / stride
//         // produces a whole number.
//         size_t num = inputWidth - filterSize + 2 * zeroPadding;
//         if (num % stride != 0)
//         {
//             cerr << "Error: Stride = " << stride << " does not evenly divide "
//                 << num << endl;
//             throw Ex("Unable to create convolutional layer.");
//         }
//     }
//
//     // Used for evaluating this layer. This updates the activation based
//     // on y = W*x + b, where * represents convolution of the inputs with
//     // each of the filters
//     void eval(const vector<T>& x) override
//     {
//         const Tensor3D<T> input((vector<T>&) x, 0, mInputWidth, mInputHeight, mInputChannels);
//
//         // Wrap the important vectors in Tensor3D objects so we can work with them
//         Tensor3D<T> output(mActivation, 0, mOutputWidth, mOutputHeight, mNumFilters);
//
//         // Note: These loops can be run in any order. Each iteration is completely
//         // independent of every other iteration.
//         for (size_t filter = 0; filter < mNumFilters; ++filter)
//         {
//             for (size_t i = 0; i < mOutputHeight; ++i)
//             {
//                 for (size_t j = 0; j < mOutputWidth; ++j)
//                 {
//                     // Dot product input about (j, i) with filter 'filter'
//                     T sum = convolve(input, j, i, filter);
//                     output.set(j, i, filter, sum);
//                 }
//             }
//         }
//     }
//
//     void calculateDeltas(const vector<T>& /*x*/, T* destination) override
//     {
//         std::fill(destination, destination + getNumParameters(), T{});
//         size_t numFilterWeights = mFilterSize * mFilterSize * mInputChannels + 1;
//
//         // Wrap the important vectors in Tensor3D objects so access is easier
//         Tensor3D<T> currentDeltas(mDeltas, 0, mOutputWidth, mOutputHeight, mNumFilters);
//         Tensor3D<T> targetDeltas(destination, 0, mInputWidth, mInputHeight, mInputChannels);
//
//         // Convolve the filters with the deltas for this layer to get the
//         // deltas for the next layer downstream (to the left)
//         for (size_t k = 0; k < mNumFilters; ++k)
//         {
//             // Wrap the current filter in a Tensor3D object
//             Tensor3D<T> currentFilter(mParameters + (k * numFilterWeights),
//                 0, mFilterSize, mFilterSize, mInputChannels);
//
//             for (size_t j = 0; j < mOutputHeight; ++j)
//             {
//                 int inRowOffset = j * mStride - mZeroPadding;
//                 for (size_t i = 0; i < mOutputWidth; ++i)
//                 {
//                     int inColOffset = i * mStride - mZeroPadding;
//                     for (size_t z = 0; z < mInputChannels; ++z)
//                     {
//                         for (size_t y = 0; y < mFilterSize; ++y)
//                         {
//                             int inRow = inRowOffset + y;
//                             for (size_t x = 0; x < mFilterSize; ++x)
//                             {
//                                 int inCol = inColOffset + x;
//                                 if (inRow >= 0 && inRow < (int)mInputHeight && inCol >= 0 && inCol < (int) mInputWidth)
//                                 {
//                                     T val = currentFilter.get(x, y, z) * currentDeltas.get(i, j, k);
//                                     targetDeltas.add(inCol, inRow, z, val);
//                                 }
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//     }
//
//     void calculateGradient(const vector<T>& x, T* gradient) override
//     {
//         const size_t numFilterWeights = mFilterSize * mFilterSize * mInputChannels + 1;
//
//         // Wrap the important vectors in Tensor3D objects so access is easier
//         const Tensor3D<T> input((vector<T>&) x, 0, mInputWidth, mInputHeight, mInputChannels);
//         const Tensor3D<T> deltas(mDeltas, 0, mOutputWidth, mOutputHeight, mNumFilters);
//
//         // Convolve the deltas in this layer with the input in order to calculate
//         // the gradient for the weights in the filters. We also calculate the gradient
//         // with respect to the bias terms along the way. g(bias) = sum(deltas)
//         for (size_t k = 0; k < mNumFilters; ++k)
//         {
//             Tensor3D<T> grad(gradient, k * numFilterWeights, mFilterSize, mFilterSize, mInputChannels);
//
//             size_t outChannelOffset = k * mOutputHeight * mOutputWidth;
//             for (size_t j = 0; j < mOutputHeight; ++j)
//             {
//                 size_t outRowOffset = j * mOutputWidth;
//                 int inRowOffset     = j * mStride - mZeroPadding;
//                 for (size_t i = 0; i < mOutputWidth; ++i)
//                 {
//                     size_t index    = outChannelOffset + outRowOffset + i;
//                     int inColOffset = i * mStride - mZeroPadding;
//
//                     // Calculate the gradient of the bias for this filter
//                     gradient[(k + 1) * numFilterWeights - 1] += mDeltas[index];
//
//                     // Calculate the gradient of the weights for this filter
//                     for (size_t z = 0; z < mInputChannels; ++z)
//                     {
//                         for (size_t y = 0; y < mFilterSize; ++y)
//                         {
//                             int inRow = inRowOffset + y;
//                             for (size_t x = 0; x < mFilterSize; ++x)
//                             {
//                                 int inCol = inColOffset + x;
//                                 if (inRow >= 0 && inRow < (int) mInputHeight && inCol >= 0 && inCol < (int) mInputWidth)
//                                 {
//                                     T val = deltas.get(i, j, k) * input.get(inCol, inRow, z);
//                                     grad.add(x, y, z, val);
//                                 }
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//     }
//
//     size_t getNumParameters() const override
//     {
//         size_t filterParams = mFilterSize * mFilterSize * mInputChannels + 1;
//         return filterParams * mNumFilters;
//     }
//
//     // Adjusts all of the kernels such that they sum to 1.0.
//     void normalizeKernels()
//     {
//         T* params                 = *mParameters;
//         const size_t filterParams = mFilterSize * mFilterSize * mInputChannels + 1;
//
//         for (size_t filter = 0; filter < mNumFilters; ++filter)
//         {
//             T sum{};
//             for (size_t j = 0; j < filterParams - 1; ++j)
//             {
//                 T val = params[j];
//                 sum += val * val;
//             }
//
//             T invMagnitude = 1.0 / sqrt(sum);
//             for (size_t j = 0; j < filterParams - 1; ++j)
//                 params[j] *= invMagnitude;
//
//             params += filterParams;
//         }
//     }
//
//     // Get information specific to convolutional layers
//     size_t getOutputWidth()   { return mOutputWidth;   }
//     size_t getOutputHeight()  { return mOutputHeight;  }
//     size_t getInputWidth()    { return mInputWidth;    }
//     size_t getInputHeight()   { return mInputHeight;   }
//     size_t getInputChannels() { return mInputChannels; }
//     size_t getFilterSize()    { return mFilterSize;    }
//     size_t getNumFilters()    { return mNumFilters;    }
//     size_t getStride()        { return mStride;        }
//     size_t getZeroPadding()   { return mZeroPadding;   }
//
// private:
//     size_t mInputWidth, mInputHeight, mInputChannels;
//     size_t mFilterSize, mNumFilters;
//     size_t mStride, mZeroPadding;
//     size_t mOutputWidth, mOutputHeight;
//
//     // (x, y, z) specifies coordinates of the output cell we are working on
//     T convolve(const Tensor3D<T>& input, size_t x, size_t y, size_t z)
//     {
//         // Calculate where the weights and bias values are for this filter
//         const size_t numFilterWeights = mFilterSize * mFilterSize * mInputChannels;
//         const size_t weightsIndex     = z * (numFilterWeights + 1);
//         const size_t biasIndex        = weightsIndex + numFilterWeights;
//         const Tensor3D<T> filterWeights(mParameters, weightsIndex,
//             mFilterSize, mFilterSize, mInputChannels);
//
//         // Calculate the bounds of the input window
//         // The amount of padding dictates where the top left corner will start. To
//         // that, we add the product of the index and the stride to move the box
//         // either horizontally or vertically. The right and bottom bounds are found
//         // by adding the filter size to the left/top bounds.
//         const int lowX = -mZeroPadding + x * mStride;
//         const int lowY = -mZeroPadding + y * mStride;
//         const int hiX  = lowX + mFilterSize - 1;
//         const int hiY  = lowY + mFilterSize - 1;
//
//         // Do the actual convolution. Tensor3D objects will return 0 when the
//         // provided index is out of bounds, which is used to implement the zero
//         // padding.
//         T sum{};
//         for (size_t k = 0; k < mInputChannels; ++k)
//         {
//             for (int j = lowY; j <= hiY; ++j)
//             {
//                 for (int i = lowX; i <= hiX; ++i)
//                     sum += input.get(i, j, k) *
//                         filterWeights.get(i - lowX, j - lowY, k);
//             }
//         }
//
//         // Factor in the bias
//         return sum + mParameters[biasIndex];
//     }
// };

}
#endif

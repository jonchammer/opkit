/*
 * File:   Layer.h
 * Author: Jon C. Hammer
 *
 * Created on September 8, 2016, 5:26 PM
 */

#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <iostream>
#include "Tensor3D.h"
#include "SparseMatrixWrapper.h"
#include "ActivationFunction.h"
#include "Error.h"
#include "Acceleration.h"
#include "PrettyPrinter.h"

using std::vector;
using std::cout;
using std::cerr;
using std::endl;

namespace opkit
{

// This is the base class from which all Layers derive. The template argument
// T specifies the type used for mathematical operations (e.g. float or double).
template <class T>
class Layer
{
public:
    Layer(const size_t inputs, const size_t outputs) :
        mParameters(nullptr),
        mInputs(inputs), mOutputs(outputs),
        mActivation(outputs), mDeltas(outputs) {}

    // When eval is called with two arguments, we evaluate like normal, but also
    // copy the result into y
    void eval(const vector<T>& x, vector<T>& y)
    {
        eval(x);
        vCopy(mActivation.data(), y.data(), mOutputs);
    }

    // 1. Feedforward step
    // When eval is called with just one argument, it is assumed that the
    // result will be placed in mActivation.
    virtual void eval(const vector<T>& x) = 0;

    // 2. Calculate blame terms for each node in the previous layer
    virtual void calculateDeltas(const vector<T>& x, T* destination) = 0;

    // 3. Calculate the gradient with respect to the parameters
    virtual void calculateGradient(const vector<T>& x, T* gradient) = 0;

    // Returns the number of optimizable parameters this layer uses. Some layers
    // only transform their inputs and so have 0 parameters.
    virtual size_t getNumParameters() const = 0;

    // When layers are added to the network, they are assigned a segment of the
    // network's parameters to work with. This function tells the layer which
    // segment to use for parameter storage. The layer does not own this storage.
    // It is 'leased' from the parent network.
    void assignStorage(T* parameters)
    {
        mParameters = parameters;
        onStorageAssigned();
    }

    // This method is essentially a callback that implementing Layers can use
    // if they need to know when the Layer has been assigned to a network (and
    // thus given storage parameters to use).
    virtual void onStorageAssigned() {}

    // General layer properties
    size_t getInputs() const    { return mInputs;     }
    size_t getOutputs() const   { return mOutputs;    }
    vector<T>& getActivation()  { return mActivation; }
    vector<T>& getDeltas()      { return mDeltas;     }

protected:
    T* mParameters;           // Storage used for the parameters for this layer
    size_t mInputs, mOutputs; // The dimensions of this layer (inputs -> outputs)
    vector<T> mActivation;    // The output of this layer
    vector<T> mDeltas;        // The derivative of the network with respect to
                              // each output neuron.
};

// Fully connected layers are the basis of the MLP. They perform the linear
// transformation y = W * x + b, where the W matrix and the b vector are
// optimizable parameters.
template <class T>
class FullyConnectedLayer : public Layer<T>
{
public:

    // Allows us to use the members in the base class without specifying
    // their complete names
    using Layer<T>::mParameters;
    using Layer<T>::mInputs;
    using Layer<T>::mOutputs;
    using Layer<T>::mDeltas;
    using Layer<T>::mActivation;

    // Create a new FullyConnectedLayer. We need only specify the input and
    // output dimensions.
    FullyConnectedLayer(const size_t inputs, const size_t outputs) :
        Layer<T>(inputs, outputs) {}

    void eval(const vector<T>& x) override
    {
        //Cache these so we can avoid repeated pointer dereferencing
        const T* params = mParameters;
        const T* xData  = x.data();
        T* yData        = mActivation.data();

        // y = W * x + b
        // Weights are arranged as an 'mOutputs' x 'mInputs' matrix in row-major
        // ordering, followed directly by the biases. E.g.:
        // [w11, w21, ... wn1]
        // [w12, w22, ... wn2]
        // [...  ...  ... ...]
        // [w1m, w2m, ... wnm]
        // [b1, b2,   ...  bm]
        //
        // This could be done with one BLAS call, but it would override the bias,
        // which means a copy would have to be made ahead of time. It's easier
        // to just use two calls.
        mvMultiply(params, xData, yData, mOutputs, mInputs);
        vAdd(params + (mInputs * mOutputs), yData, mOutputs);
    }

    void calculateDeltas(const vector<T>& /*x*/, T* destination) override
    {
        const T* deltas = mDeltas.data();

        // Calculate destination = W^T * deltas
        mtvMultiply(mParameters, deltas, destination, mOutputs, mInputs);
    }

    void calculateGradient(const vector<T>& x, T* gradient) override
    {
        // Cache the raw pointer so we can avoid calling vector::operator[]
        const T* input  = x.data();
        const T* deltas = mDeltas.data();

        // Calculate gradient for the weights and for the biases.
        // The gradients are added to the values already present.
        // gradient(weights) = outerProduct(x, deltas);
        // gradient(biases)  = deltas
        outerProduct(deltas, input, gradient, mOutputs, mInputs);
        vAdd(deltas, gradient + (mInputs * mOutputs), mOutputs);
    }

    size_t getNumParameters() const override
    {
        // N * M for the weights matrix and M for the bias terms
        return mInputs * mOutputs + mOutputs;
    }
};

// This layer performs a simple element-wise transformation to the inputs.
// Therefore the input and output sizes are the same, and this layer has no
// optimizable parameters.
template <class T>
class ActivationLayer : public Layer<T>
{
public:

    // Allows us to use the members in the base class without specifying
    // their complete names
    using Layer<T>::mOutputs;
    using Layer<T>::mActivation;
    using Layer<T>::mDeltas;

    // Create a new layer with the given size and transformation function.
    ActivationLayer(const size_t size, Activation<T>* activation, bool ownActivation = true) :
        Layer<T>(size, size),
        mActivationFunction(activation),
        mOwnActivation(ownActivation)
    {}

    ~ActivationLayer()
    {
        // If we own the pointer, we need to take charge of disposing of it.
        if (mOwnActivation)
        {
            delete mActivationFunction;
            mActivationFunction = nullptr;
        }
    }

    // Performs the element-wise transformation according to the given
    // transformation function.
    void eval(const vector<T>& x) override
    {
        const T* xData = x.data();
        T* yData       = mActivation.data();

        for (size_t i = 0; i < mOutputs; ++i)
            yData[i] = mActivationFunction->eval(xData[i]);
    }

    // The deltas for the downstream (left) layer are simply the deltas from
    // this layer multiplied by the derivative of the transformation function.
    void calculateDeltas(const vector<T>& x, T* destination) override
    {
        const T* input      = x.data();
        const T* deltas     = mDeltas.data();
        const T* activation = mActivation.data();

        for (size_t i = 0; i < mOutputs; ++i)
            destination[i] = deltas[i] * mActivationFunction->deriv(input[i], activation[i]);
    }

    void calculateGradient(const vector<T>& x, T* gradient) override
    {
        // We have no parameters, so there is no gradient to calculate
        // for this layer.
    }

    size_t getNumParameters() const override
    {
        return 0;
    }

private:
    Activation<T>* mActivationFunction;
    bool mOwnActivation;
};

// Fundamentally, SparseLayer is exceptionally similar to FullyConnectedLayer.
// The only difference is that this class uses sparse storage to reduce the
// computational complexity of the evaluation and backprop steps.
template <class T>
class SparseLayer : public Layer<T>
{
public:

    // Allows us to use the members in the base class without specifying
    // their complete names
    using Layer<T>::mParameters;
    using Layer<T>::mInputs;
    using Layer<T>::mOutputs;
    using Layer<T>::mDeltas;
    using Layer<T>::mActivation;

    SparseLayer(const size_t inputs, const size_t outputs,
        const double fillPercentage, Rand& rand) :
        Layer<T>(inputs, outputs),
        mNumConnections(size_t(fillPercentage * inputs * outputs))
    {
        // Assign rows and columns for each connection
        if (outputs * inputs == mNumConnections)
            mWeights = new SparseMatrixWrapper<T>(outputs, inputs);

        // Fill the connections randomly
        else
        {
            mWeights = new SparseMatrixWrapper<T>(outputs, inputs,
                mNumConnections, rand);
        }
    }

    ~SparseLayer()
    {
        delete mWeights;
        mWeights = nullptr;
    }

    void eval(const vector<T>& x) override
    {
        // Calculate activation = W * x + b
        T* yData = mActivation.data();
        mWeights->multiply(x.data(), yData);
        vAdd(mParameters + mNumConnections, yData, mOutputs);
    }

    void calculateDeltas(const vector<T>& /*x*/, T* destination) override
    {
        // Calculate destination = W^T * deltas
        mWeights->multiplyTranspose(mDeltas.data(), destination);
    }

    void calculateGradient(const vector<T>& x, T* gradient) override
    {
        // gradient += deltas * x^T
        const T* deltas = mDeltas.data();
        mWeights->outerProduct(deltas, x.data(), gradient);
        vAdd(deltas, gradient + mNumConnections, mOutputs);
    }

    size_t getNumParameters() const override
    {
        return mNumConnections + mOutputs;
    }

    void onStorageAssigned() override
    {
        // As soon as this layer is assigned storage, we will need to sync the
        // parameters to the sparse weights matrix. (This might happen multiple
        // times as multiple layers are added to the network).
        mWeights->setData(mParameters);
    }

    // Simple getters
    size_t getNumConnections()           const { return mNumConnections; }
    SparseMatrixWrapper<T>* getWeights() const { return mWeights;        }

private:
    size_t mNumConnections;
    SparseMatrixWrapper<T>* mWeights;
};

// Convolutional layers are often used for image processing. They take as input
// a 3D volume of numbers and produce as output another 3D volume. Typically,
// the three dimensions will correspond to the width of an image, the height of
// an image, and the number of channels in the image. Convolutional layers use
// weights that are connected to a small area, in addition to weight sharing.
// A set of 1 or more kernels are what are learned during training.
template <class T>
class Convolutional2DLayer : public Layer<T>
{
public:

    // Allows us to use the members in the base class without specifying
    // their complete names
    using Layer<T>::mParameters;
    using Layer<T>::mInputs;
    using Layer<T>::mOutputs;
    using Layer<T>::mActivation;
    using Layer<T>::mDeltas;

    // Constructors

    // Create a 2D convolutional layer with the given metaparameters.
    //
    // inputWidth    - The width of the input image in pixels
    // inputHeight   - The height of the input image in pixels
    // inputChannels - 1 for grayscale, 3 for RGB, etc.
    // filterSize    - The size of the convolution kernel (e.g. 3 or 5)
    // numFilters    - How many filters should be applied to the input
    // stride        - Allows user to adjust how much the receptive fields
    //                 overlap with one another. A value of 1 is always allowed,
    //                 but other values can cause problems if they don't
    //                 partition the space evenly.
    // zeroPadding   - The input is padded with 0 or more 0's in order to adjust
    //                 the size of the output.
    //
    // NOTE 1:
    // When it is desired to have an output size that matches the input
    // size, the padding can be calculated using the following formula:
    //   P = (SW - S - W + F) / 2, where
    // S = stride, W = inputWidth/inputHeight, F = filterSize
    //
    // NOTE 2:
    // The input volume will be of size:
    //   inputWidth * inputHeight * inputChannels.
    // The output volume will be of size:
    //   outputWidth * outputHeight * numFilters
    // where
    // outputWidth  = (inputWidth  - numFilters + 2 * zeroPadding) / stride + 1
    // outputHeight = (inputHeight - numFilters + 2 * zeroPadding) / stride + 1
    Convolutional2DLayer(
        size_t inputWidth, size_t inputHeight, size_t inputChannels,
        size_t filterSize, size_t numFilters, size_t stride, size_t zeroPadding) :

        // Superclass constructor - input and output dimensions
        Layer<T>(inputWidth * inputHeight * inputChannels,
            ((inputWidth - filterSize + 2 * zeroPadding) / stride + 1) *
            ((inputHeight - filterSize + 2 * zeroPadding) / stride + 1) *
            numFilters),

        // Paramaters
        mInputWidth(inputWidth), mInputHeight(inputHeight),
        mInputChannels(inputChannels), mFilterSize(filterSize),
        mNumFilters(numFilters), mStride(stride), mZeroPadding(zeroPadding),
        mOutputWidth((inputWidth - filterSize + 2 * zeroPadding) / stride + 1),
        mOutputHeight((inputHeight - filterSize + 2 * zeroPadding) / stride + 1)
    {
        if (stride < 1)
        {
            cerr << "Error: Stride must be at least 1." << endl;
            throw Ex("Unable to create convolutional layer.");
        }

        // Check to make sure the given metaparameters are actually a valid
        // configuration. We have to make sure that this equation:
        // (inputWidth - filterSize + (2 * zeroPadding)) / stride
        // produces a whole number.
        size_t num = inputWidth - filterSize + 2 * zeroPadding;
        if (num % stride != 0)
        {
            cerr << "Error: Stride = " << stride << " does not evenly divide "
                << num << endl;
            throw Ex("Unable to create convolutional layer.");
        }
    }

    // Used for evaluating this layer. This updates the activation based
    // on y = W*x + b, where * represents convolution of the inputs with
    // each of the filters
    void eval(const vector<T>& x) override
    {
        const Tensor3D<T> input((vector<T>&) x, 0, mInputWidth, mInputHeight, mInputChannels);

        // Wrap the important vectors in Tensor3D objects so we can work with them
        Tensor3D<T> output(mActivation, 0, mOutputWidth, mOutputHeight, mNumFilters);

        // Note: These loops can be run in any order. Each iteration is completely
        // independent of every other iteration.
        for (size_t filter = 0; filter < mNumFilters; ++filter)
        {
            for (size_t i = 0; i < mOutputHeight; ++i)
            {
                for (size_t j = 0; j < mOutputWidth; ++j)
                {
                    // Dot product input about (j, i) with filter 'filter'
                    T sum = convolve(input, j, i, filter);
                    output.set(j, i, filter, sum);
                }
            }
        }
    }

    void calculateDeltas(const vector<T>& /*x*/, T* destination) override
    {
        std::fill(destination, destination + getNumParameters(), T{});
        size_t numFilterWeights = mFilterSize * mFilterSize * mInputChannels + 1;

        // Wrap the important vectors in Tensor3D objects so access is easier
        Tensor3D<T> currentDeltas(mDeltas, 0, mOutputWidth, mOutputHeight, mNumFilters);
        Tensor3D<T> targetDeltas(destination, 0, mInputWidth, mInputHeight, mInputChannels);

        // Convolve the filters with the deltas for this layer to get the
        // deltas for the next layer downstream (to the left)
        for (size_t k = 0; k < mNumFilters; ++k)
        {
            // Wrap the current filter in a Tensor3D object
            Tensor3D<T> currentFilter(mParameters + (k * numFilterWeights),
                0, mFilterSize, mFilterSize, mInputChannels);

            for (size_t j = 0; j < mOutputHeight; ++j)
            {
                int inRowOffset = j * mStride - mZeroPadding;
                for (size_t i = 0; i < mOutputWidth; ++i)
                {
                    int inColOffset = i * mStride - mZeroPadding;
                    for (size_t z = 0; z < mInputChannels; ++z)
                    {
                        for (size_t y = 0; y < mFilterSize; ++y)
                        {
                            int inRow = inRowOffset + y;
                            for (size_t x = 0; x < mFilterSize; ++x)
                            {
                                int inCol = inColOffset + x;
                                if (inRow >= 0 && inRow < (int)mInputHeight && inCol >= 0 && inCol < (int) mInputWidth)
                                {
                                    T val = currentFilter.get(x, y, z) * currentDeltas.get(i, j, k);
                                    targetDeltas.add(inCol, inRow, z, val);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void calculateGradient(const vector<T>& x, T* gradient) override
    {
        const size_t numFilterWeights = mFilterSize * mFilterSize * mInputChannels + 1;

        // Wrap the important vectors in Tensor3D objects so access is easier
        const Tensor3D<T> input((vector<T>&) x, 0, mInputWidth, mInputHeight, mInputChannels);
        const Tensor3D<T> deltas(mDeltas, 0, mOutputWidth, mOutputHeight, mNumFilters);

        // Convolve the deltas in this layer with the input in order to calculate
        // the gradient for the weights in the filters. We also calculate the gradient
        // with respect to the bias terms along the way. g(bias) = sum(deltas)
        for (size_t k = 0; k < mNumFilters; ++k)
        {
            Tensor3D<T> grad(gradient, k * numFilterWeights, mFilterSize, mFilterSize, mInputChannels);

            size_t outChannelOffset = k * mOutputHeight * mOutputWidth;
            for (size_t j = 0; j < mOutputHeight; ++j)
            {
                size_t outRowOffset = j * mOutputWidth;
                int inRowOffset     = j * mStride - mZeroPadding;
                for (size_t i = 0; i < mOutputWidth; ++i)
                {
                    size_t index    = outChannelOffset + outRowOffset + i;
                    int inColOffset = i * mStride - mZeroPadding;

                    // Calculate the gradient of the bias for this filter
                    gradient[(k + 1) * numFilterWeights - 1] += mDeltas[index];

                    // Calculate the gradient of the weights for this filter
                    for (size_t z = 0; z < mInputChannels; ++z)
                    {
                        for (size_t y = 0; y < mFilterSize; ++y)
                        {
                            int inRow = inRowOffset + y;
                            for (size_t x = 0; x < mFilterSize; ++x)
                            {
                                int inCol = inColOffset + x;
                                if (inRow >= 0 && inRow < (int) mInputHeight && inCol >= 0 && inCol < (int) mInputWidth)
                                {
                                    T val = deltas.get(i, j, k) * input.get(inCol, inRow, z);
                                    grad.add(x, y, z, val);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    size_t getNumParameters() const override
    {
        size_t filterParams = mFilterSize * mFilterSize * mInputChannels + 1;
        return filterParams * mNumFilters;
    }

    // Adjusts all of the kernels such that they sum to 1.0.
    void normalizeKernels()
    {
        T* params                 = *mParameters;
        const size_t filterParams = mFilterSize * mFilterSize * mInputChannels + 1;

        for (size_t filter = 0; filter < mNumFilters; ++filter)
        {
            T sum{};
            for (size_t j = 0; j < filterParams - 1; ++j)
            {
                T val = params[j];
                sum += val * val;
            }

            T invMagnitude = 1.0 / sqrt(sum);
            for (size_t j = 0; j < filterParams - 1; ++j)
                params[j] *= invMagnitude;

            params += filterParams;
        }
    }

    // Get information specific to convolutional layers
    size_t getOutputWidth()   { return mOutputWidth;   }
    size_t getOutputHeight()  { return mOutputHeight;  }
    size_t getInputWidth()    { return mInputWidth;    }
    size_t getInputHeight()   { return mInputHeight;   }
    size_t getInputChannels() { return mInputChannels; }
    size_t getFilterSize()    { return mFilterSize;    }
    size_t getNumFilters()    { return mNumFilters;    }
    size_t getStride()        { return mStride;        }
    size_t getZeroPadding()   { return mZeroPadding;   }

private:
    size_t mInputWidth, mInputHeight, mInputChannels;
    size_t mFilterSize, mNumFilters;
    size_t mStride, mZeroPadding;
    size_t mOutputWidth, mOutputHeight;

    // (x, y, z) specifies coordinates of the output cell we are working on
    T convolve(const Tensor3D<T>& input, size_t x, size_t y, size_t z)
    {
        // Calculate where the weights and bias values are for this filter
        const size_t numFilterWeights = mFilterSize * mFilterSize * mInputChannels;
        const size_t weightsIndex     = z * (numFilterWeights + 1);
        const size_t biasIndex        = weightsIndex + numFilterWeights;
        const Tensor3D<T> filterWeights(mParameters, weightsIndex,
            mFilterSize, mFilterSize, mInputChannels);

        // Calculate the bounds of the input window
        // The amount of padding dictates where the top left corner will start. To
        // that, we add the product of the index and the stride to move the box
        // either horizontally or vertically. The right and bottom bounds are found
        // by adding the filter size to the left/top bounds.
        const int lowX = -mZeroPadding + x * mStride;
        const int lowY = -mZeroPadding + y * mStride;
        const int hiX  = lowX + mFilterSize - 1;
        const int hiY  = lowY + mFilterSize - 1;

        // Do the actual convolution. Tensor3D objects will return 0 when the
        // provided index is out of bounds, which is used to implement the zero
        // padding.
        T sum{};
        for (size_t k = 0; k < mInputChannels; ++k)
        {
            for (int j = lowY; j <= hiY; ++j)
            {
                for (int i = lowX; i <= hiX; ++i)
                    sum += input.get(i, j, k) *
                        filterWeights.get(i - lowX, j - lowY, k);
            }
        }

        // Factor in the bias
        return sum + mParameters[biasIndex];
    }
};

// This layer implements the multivariate logistic function (aka Softmax). Given
// a vector of inputs, it produces a vector of outputs such that the sum of the
// values is equal to 1.0. This makes softmax layers good choices when we need
// to predict a probability distribution.
template <class T>
class SoftmaxLayer : public Layer<T>
{
public:

    // Allows us to use the members in the base class without specifying
    // their complete names
    using Layer<T>::mOutputs;
    using Layer<T>::mDeltas;
    using Layer<T>::mActivation;

    // Create a Softmax layer. All we need to know is the dimension.
    SoftmaxLayer(size_t size) : Layer<T>(size, size) {}

    void eval(const vector<T>& x) override
    {
        const T* xData = x.data();
        T* yData       = mActivation.data();

        // Calculate the offset (for numerical stability)
        T offset = -xData[vMaxIndex(xData, mOutputs)];

        // y_i = e^(x_i)
        T sum{};
        for (size_t i = 0; i < mOutputs; ++i)
        {
            yData[i] = exp(xData[i] + offset);
            sum     += yData[i];
        }

        // Normalize the entries such that they sum to 1.0
        vScale(yData, 1.0 / sum, mOutputs);
    }

    void calculateDeltas(const vector<T>& /*x*/, T* destination) override
    {
        static vector<T> jacobian(mOutputs * mOutputs);
        const T* deltas     = mDeltas.data();
        const T* activation = mActivation.data();
        T* work             = jacobian.data();

        // Destination = J * deltas, where J = Diag(y) - y*y^T
        // and y is the activation of this layer. We construct J manually by
        // first calculating y*y^T (using the outerProduct function), and then
        // adding the diagonal terms.
        //
        // NOTE 1: J should be a symmetric matrix, so when possible, we should
        // avoid repeated calculations.
        //
        // NOTE 2: J can also be expressed as
        // J_ij = y_i (delta_i_j - y_j), where delta_i_j == 1 if i == j and
        // delta_i_j == 0 if i != j.
        // This definition is more useful if the matrix has to be constructed
        // manually.
        //
        // NOTE 3: If the Cross-Entropy cost function is used and this is the
        // last layer of the network, the values placed in 'destination' should
        // equal y' - y, where y' is the activation of this layer and y is the
        // training label (in one hot representation). This is computationally
        // much easier to calculate and much more numerically stable, so it
        // should be used whenever possible.
        std::fill(jacobian.begin(), jacobian.end(), T{});
        outerProduct(activation, activation, work, mOutputs, mOutputs, T{-1.0});

        size_t index = 0;
        for (size_t i = 0; i < mOutputs; ++i)
        {
            work[index] += activation[i];
            index       += mOutputs + 1;
        }

        symmetricMvMultiply(work, deltas, destination, mOutputs);
    }

    void calculateGradient(const vector<T>& x, T* gradient) override
    {
        // We have no parameters, so there is no gradient to calculate
        // for this layer.
    }

    size_t getNumParameters() const override
    {
        return 0;
    }
};

};

#endif /* LAYER_H */

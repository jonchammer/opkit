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
#include "ActivationFunction.h"
#include "Error.h"

using std::vector;
using std::cerr;
using std::endl;

namespace opkit
{

// A neural network consists of a set of layers. All layers have some functionality
// in common, so they all derive from this superclass.
template <class T>
class Layer
{
public:

    // Process the given input according to the rules for this layer, and place
    // the results in 'y'.
    virtual inline void feed(const vector<T>& x)
    {
        feed(x, getActivation());
    }
    
    virtual void feed(const vector<T>& x, vector<T>& y) = 0;
    
    // In order to efficiently calculate the gradient of a network, we need to
    // be able to assign 'blame' to each node. This method calculates the
    // 'blame' terms for each node in the previous (left) layer. Therefore, the
    // destination will usually be the deltas of the previous layer. 
    //
    // This method can also be used to calculate the gradient with respect to the 
    // inputs. In this case, the method will be called on the first layer in the
    // network, and the destination will be a location to store the gradient.
    virtual void calculateDeltas(vector<T>& destination) = 0;

    // Multiplies a single delta value by the derivative of the activation function 
    // in order to deactivate it. (Used in the output layer)
    virtual void deactivateDelta(size_t outputIndex);

    // Multiplies the deltas by the derivative of the activation function in
    // order to deactivate them. (Used in the inner and output layers)
    virtual void deactivateDeltas();

    // Calculate the gradient of the network with respect to the parameters
    // of this layer. The caller can assume that 'gradient' is a region of
    // contiguous memory that has already been allocated to be the proper size.
    // The calculated gradient is added to whatever value is already in that cell,
    // so make sure 'gradient' is initialized with 0's if that behavior is desired.
    virtual void calculateGradient(const vector<T>& input, T* gradient) = 0;
    
    // These functions provide structural information about the layer
    virtual size_t getNumParameters()           = 0;
    virtual size_t getInputs()                  = 0;
    virtual size_t getOutputs()                 = 0;
    virtual Activation& getActivationFunction() = 0;
    
    virtual vector<T>& getNet() = 0;
    
    // All layers produce some sort of output. This method returns that output
    // to the caller.
    virtual vector<T>& getActivation() = 0;
    
    // This method returns the deltas that were calculated in calculateDeltas()
    // to the user.
    virtual vector<T>& getDeltas() = 0;
    
    // When layers are added to the network, they are assigned a segment of the
    // network's parameters to work with. This function tells the layer which
    // segment to use for parameter storage.
    void assignStorage(vector<T>* parameters, const size_t parametersStartIndex);
    
protected:
    vector<T>* mParameters;  // Parameter storage
    size_t mParametersStartIndex;
};

// Feedforward layers are traditional, fully connected layers. They take vectors
// as input, and produce vectors as output.
template <class T>
class FeedforwardLayer : public Layer<T>
{
public:
    // Constructors
    FeedforwardLayer(size_t inputs, size_t outputs);
    
    // Used for evaluating this layer. This updates the activation based 
    // on y = a(Wx + b)
    void feed(const vector<T>& x, vector<T>& y) override;
    
    void calculateDeltas(vector<T>& destination)                override;
    void calculateGradient(const vector<T>& input, T* gradient) override;
    
    size_t getNumParameters()       override;     
    size_t getInputs()              override;    
    size_t getOutputs()             override;     
    
    vector<T>& getNet()        override;
    vector<T>& getActivation() override;
    vector<T>& getDeltas()     override;
    
    Activation& getActivationFunction() override;
    void setActivationFunction(Activation act);
 
private:
    size_t mInputs, mOutputs;   // The dimensions of this layer
    vector<T> mDeltas;          // The errors that result from backprop
    vector<T> mNet;             // The sum before the activation function is applied
    vector<T> mActivation;      // The activation (output of this layer)
    Activation mActFunction;    // The activation function (and derivative) to be used in this layer
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
    Convolutional2DLayer(size_t inputWidth, size_t inputHeight, size_t inputChannels, 
        size_t filterSize, size_t numFilters, size_t stride, size_t zeroPadding);
    virtual ~Convolutional2DLayer() {}
    
    // Used for evaluating this layer. This updates the activation based 
    // on y = a(W*x + b), where * represents convolution of the inputs with
    // each of the filters
    void feed(const vector<T>& x, vector<T>& y);
        
    void calculateDeltas(vector<T>& destination)                 override;
    void calculateGradient(const vector<T>& input, T* gradient) override;
    
    // Adjusts all of the kernels such that they sum to 1.0.
    void normalizeKernels();
    
    size_t getNumParameters()       override;     
    size_t getInputs()              override;    
    size_t getOutputs()             override;     
  
    vector<T>& getNet()        override;
    vector<T>& getActivation() override;
    vector<T>& getDeltas()     override;
    
    size_t getOutputWidth();
    size_t getOutputHeight();
    size_t getInputWidth();
    size_t getInputHeight();
    size_t getInputChannels();
    size_t getFilterSize();
    size_t getNumFilters();
    size_t getStride();
    size_t getZeroPadding();
    
    Activation& getActivationFunction() override;
    void setActivationFunction(Activation act);
 
private:
    size_t mInputWidth, mInputHeight, mInputChannels;
    size_t mFilterSize, mNumFilters;
    size_t mStride, mZeroPadding;
    size_t mOutputWidth, mOutputHeight;
    
    vector<T> mNet;        // The sum before the activation function is applied
    vector<T> mActivation; // The activation (output of this layer)
    vector<T> mDeltas;     // The errors that result from backprop
    Activation mActFunction;    // The activation function (and derivative) to be used in this layer
    
    T convolve(const Tensor3D<T>& input, size_t x, size_t y, size_t z);
    
    // Convolves a 2D input with a 2D filter in order to produce a 2D output.
    // The 'z' parameters specify which slice of the 3D Tensors to use. The
    // output size will be determined by the padding and stride values.
    // The values calculated are SUMMED to whatever is currently inside 'output'.
    void convolve(Tensor3D<T>& input,  size_t ix, size_t iy, size_t iz,
        Tensor3D<T>& filter, size_t filterZ,
        Tensor3D<T>& output, size_t outputZ);
};

// ---- Layer Implementations ---- //
template <class T>
void Layer<T>::deactivateDelta(size_t outputIndex)
{
    ActivationDeriv deriv  = *getActivationFunction().second;
    vector<T>& net         = getNet();
    vector<T>& act         = getActivation();
    vector<T>& deltas      = getDeltas();

    deltas[outputIndex] *= deriv(net[outputIndex], act[outputIndex]);
}

template <class T>
void Layer<T>::deactivateDeltas()
{
    ActivationDeriv deriv  = *getActivationFunction().second;
    vector<T>& net         = getNet();
    vector<T>& act         = getActivation();
    vector<T>& deltas      = getDeltas();
    const size_t outputs   = getOutputs();
    
    for (size_t i = 0; i < outputs; ++i)
        deltas[i] *= deriv(net[i], act[i]);
}

template <class T>
void Layer<T>::assignStorage(vector<T>* parameters, 
    const size_t parametersStartIndex)
{
    mParameters           = parameters;
    mParametersStartIndex = parametersStartIndex;
}

// ---- FeedforwardLayer Implementations ---- //

template <class T>
FeedforwardLayer<T>::FeedforwardLayer(size_t inputs, size_t outputs)
    : mInputs(inputs), mOutputs(outputs)
{
    mDeltas.resize(outputs);
    mNet.resize(outputs);
    mActivation.resize(outputs);
    
    // Use tanh() as the default activation function
    mActFunction = tanhActivation;
}
    
template <class T>
void FeedforwardLayer<T>::feed(const vector<T>& x, vector<T>& y)
{
    y.resize(mOutputs, 0.0);
    
    // Cache these so we can avoid repeated pointer dereferencing
    const T* params      = Layer<T>::mParameters->data();
    const T* xData       = x.data();
    T* net               = mNet.data();
    ActivationFunc func  = *getActivationFunction().first;
    
    // Cache the beginning of the bias section of the parameters
    const size_t biasStart = mInputs * mOutputs;
    
    // The weights are arranged so they can be examined in order
    size_t weightIndex = 0;

    for (size_t j = 0; j < mOutputs; ++j)
    {
        // sum = W * x
        T val = 0.0;
 
        for (size_t i = 0; i < mInputs; ++i)
        {
            val += params[Layer<T>::mParametersStartIndex + weightIndex] * xData[i];
            weightIndex++;
        }
        val += params[Layer<T>::mParametersStartIndex + biasStart + j];
        net[j] = val;
        
        // y = a(sum + bias)
        y[j] = func(val);
        
        // Usually, y will be mActivation, but there's no need for that to be
        // the case. If not, we need to make sure to set mActivation manually,
        // since some algorithms depend on it.
        mActivation[j] = y[j];
    }
}

template <class T>
void FeedforwardLayer<T>::calculateDeltas(vector<T>& destination)
{
    std::fill(destination.begin(), destination.end(), 0.0);
    const vector<T>& params = *Layer<T>::mParameters;
    
    for (size_t i = 0; i < mInputs; ++i)
    {
        for (size_t j = 0; j < mOutputs; ++j)
        {
            // params[index] is the current weight
            size_t index    = Layer<T>::mParametersStartIndex + (j * mInputs + i);
            destination[i] += params[index] * mDeltas[j];
        }
    }
}

template <class T>
void FeedforwardLayer<T>::calculateGradient(const vector<T>& input, T* gradient)
{
    // Cache the raw pointer so we can avoid calling vector::operator[]
    const T* x      = input.data();
    const T* deltas = mDeltas.data();
    
    // Calculate gradient for the weights
    size_t index = 0;
    for (size_t i = 0; i < mOutputs; ++i)
    {
        for (size_t j = 0; j < mInputs; ++j)
            gradient[index++] += x[j] * deltas[i];
    }

    // Calculate gradient for the biases
    for (size_t i = 0; i < mOutputs; ++i)
        gradient[index++] += deltas[i];
}

template <class T>
size_t FeedforwardLayer<T>::getNumParameters()       
{ 
    return (mInputs + 1) * mOutputs; 
}

template <class T>
size_t FeedforwardLayer<T>::getInputs()               
{ 
    return mInputs;                  
}

template <class T>
size_t FeedforwardLayer<T>::getOutputs()              
{ 
    return mOutputs;                 
}

template <class T>
vector<T>& FeedforwardLayer<T>::getActivation()  
{ 
    return mActivation;              
}

template <class T>   
vector<T>& FeedforwardLayer<T>::getDeltas()      
{ 
    return mDeltas;      
}

template <class T>
vector<T>& FeedforwardLayer<T>::getNet()        
{ 
    return mNet;        
}

template <class T>
Activation& FeedforwardLayer<T>::getActivationFunction()         
{ 
    return mActFunction; 
}

template <class T>
void FeedforwardLayer<T>::setActivationFunction(Activation act) 
{ 
    mActFunction = act;  
}

// ---- Convolutional2DLayer Implementations ---- //
template <class T>
Convolutional2DLayer<T>::Convolutional2DLayer(size_t inputWidth, size_t inputHeight, size_t inputChannels, 
    size_t filterSize, size_t numFilters, size_t stride, size_t zeroPadding)
    : mInputWidth(inputWidth), mInputHeight(inputHeight), 
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
    
    // Ensure the vectors are big enough to hold the data
    size_t outputSize = mOutputWidth * mOutputHeight * mNumFilters;
    mNet.resize(outputSize);
    mActivation.resize(outputSize);
    mDeltas.resize(outputSize);
    
    // Use tanh() as the default activation function
    mActFunction = tanhActivation;
}

// (x, y, z) specifies coordinates of the output cell we are working on
template <class T>
T Convolutional2DLayer<T>::convolve(const Tensor3D<T>& input, size_t x, size_t y, size_t z)
{
    // Calculate where the weights and bias values are for this filter
    const size_t numFilterWeights = mFilterSize * mFilterSize * mInputChannels;
    const size_t weightsIndex     = Layer<T>::mParametersStartIndex + z * (numFilterWeights + 1);
    const size_t biasIndex        = weightsIndex + numFilterWeights;
    const Tensor3D<T> filterWeights(*Layer<T>::mParameters, weightsIndex, 
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
    T sum = 0.0;
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
    return sum + (*Layer<T>::mParameters)[biasIndex];
}

template <class T>
void Convolutional2DLayer<T>::feed(const vector<T>& x, vector<T>& y)
{
    const Tensor3D<T> input((vector<T>&) x, 0, mInputWidth, mInputHeight, mInputChannels);
    
    // Make sure the output is the correct size
    y.resize(mOutputWidth * mOutputHeight * mNumFilters);
    
    // Wrap the important vectors in Tensor3D objects so we can work with them
    Tensor3D<T> net(mNet, 0, mOutputWidth, mOutputHeight, mNumFilters);
    Tensor3D<T> activation(mActivation, 0, mOutputWidth, mOutputHeight, mNumFilters);
    Tensor3D<T> output(y, 0, mOutputWidth, mOutputHeight, mNumFilters);
    
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
                T act = (*mActFunction.first)(sum);
                
                net.set(j, i, filter, sum);
                activation.set(j, i, filter, act);
                output.set(j, i, filter, act);
            }
        }
    }
}
    
template <class T>
void Convolutional2DLayer<T>::calculateDeltas(vector<T>& destination)
{
    std::fill(destination.begin(), destination.end(), 0.0);
    
    size_t numFilterWeights = mFilterSize * mFilterSize * mInputChannels + 1;
    
    // Wrap the important vectors in Tensor3D objects so access is easier
    Tensor3D<T> currentDeltas(mDeltas, 0, mOutputWidth, mOutputHeight, mNumFilters);
    Tensor3D<T> targetDeltas(destination, 0, mInputWidth, mInputHeight, mInputChannels);
    
    // Convolve the filters with the deltas for this layer to get the
    // deltas for the next layer downstream (to the left)
    for (size_t k = 0; k < mNumFilters; ++k)
    {
        // Wrap the current filter in a Tensor3D object
        Tensor3D<T> currentFilter(*Layer<T>::mParameters, 
            Layer<T>::mParametersStartIndex + (k * numFilterWeights), 
            mFilterSize, mFilterSize, mInputChannels);
        
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

template <class T>
void Convolutional2DLayer<T>::calculateGradient(const vector<T>& x, T* gradient)
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

template <class T>
void Convolutional2DLayer<T>::normalizeKernels()
{
    vector<T>& params         = *Layer<T>::mParameters;
    const size_t filterParams = mFilterSize * mFilterSize * mInputChannels + 1;
    size_t start              = Layer<T>::mParametersStartIndex;
    
    for (size_t filter = 0; filter < mNumFilters; ++filter)
    {
        T sum = 0.0;
        for (size_t j = 0; j < filterParams - 1; ++j)
        {
            T val = params[start + j];
            sum += val * val;
        }

        T invMagnitude = 1.0 / sqrt(sum);
        for (size_t j = 0; j < filterParams - 1; ++j)
            params[start + j] *= invMagnitude;
            
        start += filterParams;
    }
}

template <class T>
size_t Convolutional2DLayer<T>::getNumParameters()
{
    size_t filterParams = mFilterSize * mFilterSize * mInputChannels + 1;
    return filterParams * mNumFilters;
}

template <class T>
size_t Convolutional2DLayer<T>::getInputs()
{
    return mInputWidth * mInputHeight * mInputChannels;
}

template <class T>
size_t Convolutional2DLayer<T>::getOutputs()
{
    return mOutputWidth * mOutputHeight * mNumFilters;
}

template <class T>
vector<T>& Convolutional2DLayer<T>::getActivation()
{
    return mActivation;
}

template <class T>
vector<T>& Convolutional2DLayer<T>::getNet()
{
    return mNet;
}

template <class T>
vector<T>& Convolutional2DLayer<T>::getDeltas()
{
    return mDeltas;
}

template <class T>
size_t Convolutional2DLayer<T>::getOutputWidth()
{
    return mOutputWidth;
}

template <class T>
size_t Convolutional2DLayer<T>::getOutputHeight()
{
    return mOutputHeight;
}

template <class T>
size_t Convolutional2DLayer<T>::getInputWidth()
{
    return mInputWidth;
}

template <class T>
size_t Convolutional2DLayer<T>::getInputHeight()
{
    return mInputHeight;
}

template <class T>
size_t Convolutional2DLayer<T>::getInputChannels()
{
    return mInputChannels;
}

template <class T>
size_t Convolutional2DLayer<T>::getFilterSize()
{
    return mFilterSize;
}

template <class T>
size_t Convolutional2DLayer<T>::getNumFilters()
{
    return mNumFilters;
}

template <class T>
size_t Convolutional2DLayer<T>::getStride()
{
    return mStride;
}

template <class T>
size_t Convolutional2DLayer<T>::getZeroPadding()
{
    return mZeroPadding;
}
   
template <class T>   
Activation& Convolutional2DLayer<T>::getActivationFunction()         
{ 
    return mActFunction; 
}

template <class T>
void Convolutional2DLayer<T>::setActivationFunction(Activation act) 
{ 
    mActFunction = act;  
}
};

#endif /* LAYER_H */


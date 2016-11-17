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

namespace athena
{

// A neural network consists of a set of layers. All layers have some functionality
// in common, so they all derive from this superclass.
class Layer
{
public:

    // Process the given input according to the rules for this layer, and place
    // the results in 'y'.
    virtual inline void feed(const vector<double>& x)
    {
        feed(x, getActivation());
    }
    
    virtual void feed(const vector<double>& x, vector<double>& y) = 0;
    
    // In order to efficiently calculate the gradient of a network, we need to
    // be able to assign 'blame' to each node. This method calculates the
    // 'blame' terms for each node in the previous (left) layer. Therefore, the
    // destination will usually be the deltas of the previous layer. 
    //
    // This method can also be used to calculate the gradient with respect to the 
    // inputs. In this case, the method will be called on the first layer in the
    // network, and the destination will be a location to store the gradient.
    virtual void calculateDeltas(vector<double>& destination) = 0;

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
    virtual void calculateGradient(const vector<double>& input, double* gradient) = 0;
    
    // These functions provide structural information about the layer
    virtual size_t getNumParameters()           = 0;
    virtual size_t getInputs()                  = 0;
    virtual size_t getOutputs()                 = 0;
    virtual Activation& getActivationFunction() = 0;
    
    virtual vector<double>& getNet() = 0;
    
    // All layers produce some sort of output. This method returns that output
    // to the caller.
    virtual vector<double>& getActivation() = 0;
    
    // This method returns the deltas that were calculated in calculateDeltas()
    // to the user.
    virtual vector<double>& getDeltas() = 0;
    
    // When layers are added to the network, they are assigned a segment of the
    // network's parameters to work with. This function tells the layer which
    // segment to use for parameter storage.
    void assignStorage(vector<double>* parameters, 
        const size_t parametersStartIndex);
    
protected:
    vector<double>* mParameters;  // Parameter storage
    size_t mParametersStartIndex;
};

// Feedforward layers are traditional, fully connected layers. They take vectors
// as input, and produce vectors as output.
class FeedforwardLayer : public Layer
{
public:
    // Constructors
    FeedforwardLayer(size_t inputs, size_t outputs);
    
    // Used for evaluating this layer. This updates the activation based 
    // on y = a(Wx + b)
    void feed(const vector<double>& x, vector<double>& y) override;
    
    void calculateDeltas(vector<double>& destination)                     override;
    void calculateGradient(const vector<double>& input, double* gradient) override;
    
    size_t getNumParameters()       override;     
    size_t getInputs()              override;    
    size_t getOutputs()             override;     
    
    vector<double>& getNet()        override;
    vector<double>& getActivation() override;
    vector<double>& getDeltas()     override;
    
    Activation& getActivationFunction() override;
    void setActivationFunction(Activation act);
 
private:
    size_t mInputs, mOutputs;   // The dimensions of this layer
    vector<double> mDeltas;     // The errors that result from backprop
    vector<double> mNet;        // The sum before the activation function is applied
    vector<double> mActivation; // The activation (output of this layer)
    Activation mActFunction;    // The activation function (and derivative) to be used in this layer
};

// Convolutional layers are often used for image processing. They take as input
// a 3D volume of numbers and produce as output another 3D volume. Typically,
// the three dimensions will correspond to the width of an image, the height of
// an image, and the number of channels in the image. Convolutional layers use
// weights that are connected to a small area, in addition to weight sharing.
// A set of 1 or more kernels are what are learned during training.
class Convolutional2DLayer : public Layer
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
    void feed(const vector<double>& x, vector<double>& y);
        
    void calculateDeltas(vector<double>& destination)                     override;
    void calculateGradient(const vector<double>& input, double* gradient) override;
    
    // Adjusts all of the kernels such that they sum to 1.0.
    void normalizeKernels();
    
    size_t getNumParameters()       override;     
    size_t getInputs()              override;    
    size_t getOutputs()             override;     
  
    vector<double>& getNet()        override;
    vector<double>& getActivation() override;
    vector<double>& getDeltas()     override;
    
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
    
    vector<double> mNet;        // The sum before the activation function is applied
    vector<double> mActivation; // The activation (output of this layer)
    vector<double> mDeltas;     // The errors that result from backprop
    Activation mActFunction;    // The activation function (and derivative) to be used in this layer
    
    double convolve(const Tensor3D& input, size_t x, size_t y, size_t z);
    
    // Convolves a 2D input with a 2D filter in order to produce a 2D output.
    // The 'z' parameters specify which slice of the 3D Tensors to use. The
    // output size will be determined by the padding and stride values.
    // The values calculated are SUMMED to whatever is currently inside 'output'.
    void convolve(Tensor3D& input,  size_t ix, size_t iy, size_t iz,
        Tensor3D& filter, size_t filterZ,
        Tensor3D& output, size_t outputZ);
};

};

#endif /* LAYER_H */


/* 
 * File:   ConvNeuralNetwork.h
 * Author: Jon C. Hammer
 *
 * Created on August 20, 2016, 10:21 AM
 */

#ifndef CONVNEURALNETWORK_H
#define CONVNEURALNETWORK_H

#include <vector>
#include "PrettyPrinter.h"
#include "Function.h"
#include "Matrix.h"
#include "Error.h"
#include "ActivationFunction.h"
#include "Tensor3D.h"
using namespace std;

class Layer
{
public:

    // Process the given input according to the rules for this layer, and place
    // the results in 'y'.
    virtual void feed(const vector<double>& x);
    virtual void feed(const vector<double>& x, vector<double>& y) = 0;
    
    // In order to efficiently calculate the gradient of a network, we need to
    // be able to assign 'blame' to each node. This method calculates the
    // 'blame' terms for each output node in this layer. The first version of
    // this function is used for all layers but the last, while the second
    // version is only used for output layers.
    virtual void calculateDeltas(Layer* downstream)  = 0;
    virtual void calculateDeltas(size_t outputIndex) = 0;
    
    // Calculate the gradient of the network with respect to the parameters
    // of this layer. The caller can assume that 'gradient' is a region of
    // contiguous memory that has already been allocated to be the proper size.
    virtual void calculateGradient(const vector<double>& input,
        size_t outputIndex, double* gradient) = 0;
    
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

class FeedforwardLayer : public Layer
{
public:
    // Constructors
    FeedforwardLayer(size_t inputs, size_t outputs);
    
    // Used for evaluating this layer. This updates the activation based 
    // on y = a(Wx + b)
    void feed(const vector<double>& x, vector<double>& y) override;
    
    void calculateDeltas(Layer* downstream)  override;
    void calculateDeltas(size_t outputIndex) override;
    
    void calculateGradient(const vector<double>& input, 
        size_t outputIndex, double* gradient) override;
    
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

// A Neural Network consists of Layers.
class ConvLayer : public Layer
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
    ConvLayer(size_t inputWidth, size_t inputHeight, size_t inputChannels, 
        size_t filterSize, size_t numFilters, size_t stride, size_t zeroPadding);
    virtual ~ConvLayer() {}
    
    // Used for evaluating this layer. This updates the activation based 
    // on y = a(W*x + b), where * represents convolution of the inputs with
    // each of the filters
    void feed(const vector<double>& x, vector<double>& y);
    
    void calculateDeltas(Layer* downstream)  override;
    void calculateDeltas(size_t outputIndex) override;
    
    void calculateGradient(const vector<double>& input, 
        size_t outputIndex, double* gradient) override;
    
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
    
    vector<double> mNet;        // The sum before the activation function is applied
    vector<double> mActivation; // The activation (output of this layer)
    vector<double> mDeltas;     // The errors that result from backprop
    Activation mActFunction;    // The activation function (and derivative) to be used in this layer
    
    double convolve(Tensor3D& input, size_t x, size_t y, size_t z);
    
    // Convolves a 2D input with a 2D filter in order to produce a 2D output.
    // The 'z' parameters specify which slice of the 3D Tensors to use. The
    // output size will be determined by the padding and stride values.
    // The values calculated are SUMMED to whatever is currently inside 'output'.
    void convolve(Tensor3D& input,  size_t ix, size_t iy, size_t iz,
        Tensor3D& filter, size_t filterZ,
        Tensor3D& output, size_t outputZ);
};

// This is a model representing a standard feedforward Artificial Neural Network
// (ANN). A Neural Network consists of a set of neurons arranged in layers. Each
// neuron calculates a weighted sum of its inputs, applies a nonlinear activation
// function (e.g. tanh(x)), and outputs a result. The network topology can be
// adjusted in order to mimic any traditional function.
//
// When a Neural Network is created, the user provides the topology in the form
// of a vector of integers. Each number represents the number of neurons in the
// corresponding layer. So <4, 2, 6> would represent a network with 4 inputs,
// 2 nodes in the hidden layer, and 6 outputs.
class NeuralNetwork : public Function
{
public:
    // Constructors
    NeuralNetwork();
    virtual ~NeuralNetwork() {};
    
    // Layer modification
    void addLayer(Layer* layer);
    
    // Functions from the "Function" interface
    void evaluate(const vector<double>& input, vector<double>& output)          override;
    void calculateJacobianParameters(const vector<double>& x, Matrix& jacobian) override;
    void calculateJacobianInputs(const vector<double>& x, Matrix& gradient)     override;
    
    size_t getInputs()  const override;
    size_t getOutputs() const override;
    
    vector<double>& getParameters() override;
    const vector<double>& getParameters() const override;
    size_t getNumParameters() const override;
    
    // Getters / Setters
    size_t getNumLayers() const;
    Layer* getLayer(const size_t index);
    const Layer* getLayer(const size_t index) const;
    Layer* getOutputLayer();
    const Layer* getOutputLayer() const;
    
private:
    vector<double> mParameters;
    vector<Layer*> mLayers;
    
    // Gradient calculation helper functions
    void calculateDeltas(const size_t outputIndex);
    void calculateGradientFromDeltas(const vector<double>& feature,
        size_t outputIndex, vector<double>& gradient);
};

#endif /* CONVNEURALNETWORK_H */


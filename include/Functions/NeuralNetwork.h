/* 
 * File:   NeuralNetwork.h
 * Author: Jon C. Hammer
 *
 * Created on July 20, 2016, 9:32 AM
 */

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include "PrettyPrinter.h"
#include "Function.h"
#include "Matrix.h"
#include "Error.h"
#include "ActivationFunction.h"
using namespace std;

// A Neural Network consists of Layers.
class Layer
{
public:
    // Constructors
    Layer(size_t inputs, size_t outputs, 
        vector<double>& parameterStorage, int parameterStartIndex);
    virtual ~Layer() {}
    
    // Used for evaluating this layer. This updates the activation based 
    // on y = a(Wx + b)
    void feed(const vector<double>& x) { feed(x, mActivation); }
    void feed(const vector<double>& x, vector<double>& y);
    
    // Getters / Setters
    vector<double>& getBlame()      { return mBlame;      }
    vector<double>& getNet()        { return mNet;        }
    vector<double>& getActivation() { return mActivation; }
    int getInputSize()              { return mInputs;     }
    int getOutputSize()             { return mOutputs;    }
    
    Activation getActivationFunction()         { return mActFunction; }
    void setActivationFunction(Activation act) { mActFunction = act;  }
 
private:
    vector<double>& mParameters; // A reference to the parameters of the parent
    int mParameterStartIndex;    // Where in 'mParameters' our parameters start
    
    size_t mInputs, mOutputs;   // The dimensions of this layer
    vector<double> mBlame;      // The errors that result from backprop
    vector<double> mNet;        // The sum before the activation function is applied
    vector<double> mActivation; // The activation (output of this layer)
    Activation mActFunction;    // The activation function (and derivative) to be used in this layer
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
class NeuralNetwork : public StandardFunction
{
public:
    // Constructors
    NeuralNetwork(const vector<int>& layerDimensions);
    virtual ~NeuralNetwork() {};
    
    // Functions from the "Function" interface
    void evaluate(const vector<double>& input, vector<double>& output);
    void calculateJacobianParameters(const vector<double>& x, Matrix& jacobian);
    void calculateJacobianInputs(const vector<double>& x, Matrix& gradient);
    
    // Getters / Setters
    size_t getNumLayers() const;
    Layer& getLayer(const size_t index);
    const Layer& getLayer(const size_t index) const;
    Layer& getOutputLayer();
    const Layer& getOutputLayer() const;
    
private:
    vector<Layer> mLayers;
    
    // Gradient calculation helper functions
    void calculateDeltas(const size_t outputIndex);
    void calculateGradientFromDeltas(const vector<double>& feature,
        vector<double>& gradient);
};

#endif /* NEURALNETWORK_H */


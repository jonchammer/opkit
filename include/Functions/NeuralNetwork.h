/* 
 * File:   NeuralNetwork.h
 * Author: Jon C. Hammer
 *
 * Created on August 20, 2016, 10:21 AM
 */

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include "PrettyPrinter.h"
#include "Function.h"
#include "Matrix.h"
#include "Error.h"
#include "Layer.h"
using namespace std;

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
    // Create an empty Neural Network. Layers need to be added by calling
    // 'addLayer' before anything useful can be done with it.
    NeuralNetwork();
    
    // Destroy the Neural Network
    virtual ~NeuralNetwork() {};
    
    // Add a new layer to this Neural Network. Layers are added to the end of
    // the network, so add the layers from input layer to output layer. The 
    // caller of this function retains ownership of the layer--it will NOT be
    // destroyed (if necessary) by the network.
    void addLayer(Layer* layer);
    
    // Execute one forward pass through the network in order to produce an output.
    void evaluate(const vector<double>& input, vector<double>& output) override;
    
    // Calculates the Jacobian of the network with respect to the weights and
    // biases. This involves one forward pass and one backwards pass for each
    // output of the network.
    void calculateJacobianParameters(const vector<double>& x, Matrix& jacobian) override;
    
    // Calculates the Jacobian of the network with respect to the inputs. This
    // involves one forward pass and one backwards pass for each output of the 
    // network.
    void calculateJacobianInputs(const vector<double>& x, Matrix& gradient) override;
    
    // Neural networks do cache the last evaluation, so this function will 
    // always return true.
    bool cachesLastEvaluation() const override;
    
    // Returns the most recent output of the network.
    void getLastEvaluation(vector<double>& output) override;
    
    // Calculates the gradient of the network with respect to the parameters,
    // under the assumption that the deltas have already been calculated for
    // every applicable node in the network. The calculated gradient is added
    // to the value already in the appropriate cell in 'gradient', so make sure
    // the vector is initialized to 0 ahead of time if a fresh calculation is
    // desired.
    void calculateGradientParameters(const vector<double>& input, vector<double>& gradient);
    
    // Getters / Setters
    size_t getInputs()  const override;
    size_t getOutputs() const override;
    vector<double>& getParameters() override;
    const vector<double>& getParameters() const override;
    size_t getNumParameters() const override;
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
        vector<double>& gradient);
};

#endif /* NEURALNETWORK_H */


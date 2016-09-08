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
        vector<double>& gradient);
};

#endif /* NEURALNETWORK_H */


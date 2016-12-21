/*
 * File:   NeuralNetwork.h
 * Author: Jon C. Hammer
 *
 * Created on August 20, 2016, 10:21 AM
 */

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <cmath>
#include "PrettyPrinter.h"
#include "Function.h"
#include "Matrix.h"
#include "Error.h"
#include "Layer.h"

using std::vector;
using std::max;

namespace opkit
{

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
template <class T>
class NeuralNetwork : public Function<T>
{
public:
    // Create an empty Neural Network. Layers need to be added by calling
    // 'addLayer' before anything useful can be done with it.
    NeuralNetwork() {}

    // Destroy the Neural Network
    virtual ~NeuralNetwork() {};

    // Add a new layer to this Neural Network. Layers are added to the end of
    // the network, so add the layers from input layer to output layer. The
    // caller of this function retains ownership of the layer--it will NOT be
    // destroyed (if necessary) by the network.
    void addLayer(Layer<T>* layer);

    // Execute one forward pass through the network in order to produce an output.
    void evaluate(const vector<T>& input, vector<T>& output) override;

    // Calculates the Jacobian of the network with respect to the weights and
    // biases. This involves one forward pass and one backwards pass for each
    // output of the network.
    void calculateJacobianParameters(const vector<T>& x, Matrix<T>& jacobian) override;

    // Calculates the Jacobian of the network with respect to the inputs. This
    // involves one forward pass and one backwards pass for each output of the
    // network.
    void calculateJacobianInputs(const vector<T>& x, Matrix<T>& jacobian) override;

    // Neural networks do cache the last evaluation, so this function will
    // always return true.
    bool cachesLastEvaluation() const override
    {
        return true;
    }

    // Returns the most recent output of the network.
    void getLastEvaluation(vector<T>& output) override
    {
        vector<T>& lastActivation = mLayers.back()->getActivation();
        std::copy(lastActivation.begin(), lastActivation.end(), output.begin());
    }

    // Calculates the gradient of the network with respect to the parameters,
    // under the assumption that the deltas have already been calculated for
    // every applicable node in the network. The calculated gradient is added
    // to the value already in the appropriate cell in 'gradient', so make sure
    // the vector is initialized to 0 ahead of time if a fresh calculation is
    // desired.
    void calculateGradientParameters(const vector<T>& input, vector<T>& gradient);

    // Initializes the weights and biases with random values
    void initializeParameters();

    // Getters / Setters
    size_t getInputs()  const override
    {
        return mLayers.front()->getInputs();
    }

    size_t getOutputs() const override
    {
        return mLayers.back()->getOutputs();
    }

    vector<T>& getParameters() override
    {
        return mParameters;
    }

    const vector<T>& getParameters() const override
    {
        return mParameters;
    }

    size_t getNumParameters() const override
    {
        return mParameters.size();
    }

    size_t getNumLayers() const
    {
        return mLayers.size();
    }

    Layer<T>* getLayer(const size_t index)
    {
        return mLayers[index];
    }

    const Layer<T>* getLayer(const size_t index) const
    {
        return mLayers[index];
    }

    Layer<T>* getOutputLayer()
    {
        return mLayers.back();
    }

    const Layer<T>* getOutputLayer() const
    {
        return mLayers.back();
    }

private:
    vector<T> mParameters;
    vector<Layer<T>*> mLayers;

    // Gradient calculation helper functions
    void calculateDeltas(const size_t outputIndex);
    void calculateGradientFromDeltas(const vector<T>& feature, vector<T>& gradient);
};

template <class T>
void NeuralNetwork<T>::addLayer(Layer<T>* layer)
{
    // Make sure this layer is compatible with the rest of the network
    if (!mLayers.empty() && mLayers.back()->getOutputs() != layer->getInputs())
    {
        cerr << "This number of inputs to this layer must match the number of"
             << " outputs in the layer before." << endl;
        throw Ex("Unable to add layer.");
    }

    // Increase the network's storage to accommodate the new layer. Give the
    // new layer a share of the parameters to work with.
    size_t numParams = layer->getNumParameters();
    size_t origSize  = mParameters.size();
    mParameters.resize(origSize + numParams);
    mLayers.push_back(layer);

    // When the parameters vector is resized, it's possible that the pointers
    // may have been invalidated. To guard against that, we need to reassign
    // the storage for all of the layers. (Yes, this is inefficient, but it is
    // assumed that the user isn't going to be adding more layers to the network
    // at runtime.)
    T* data = mParameters.data();
    for (Layer<T>* l : mLayers)
    {
        l->assignStorage(data);
        data += l->getNumParameters();
    }
}

template <class T>
void NeuralNetwork<T>::evaluate(const vector<T>& input, vector<T>& output)
{
    // For single layer networks, we feed directly into the output
    if (mLayers.size() == 1)
        mLayers[0]->eval(input, output);

    else
    {
        // Feed the input to the first layer and put the result in Layer 1's
        // activation
        mLayers[0]->eval(input);

        // Feed the activation from the previous layer into the current layer.
        for (size_t i = 1; i < mLayers.size() - 1; ++i)
            mLayers[i]->eval(mLayers[i - 1]->getActivation());

        // On the last layer, feed the previous layer's activation and put the
        // result in 'output'
        mLayers.back()->eval(mLayers[mLayers.size() - 2]->getActivation(), output);
    }
}

template <class T>
void NeuralNetwork<T>::calculateDeltas(const size_t outputIndex)
{
    // Calculate the deltas on the last layer first
    // vector<T>& outputDeltas = mLayers.back()->getDeltas();
    // std::fill(outputDeltas.begin(), outputDeltas.end(), 0.0);
    // outputDeltas[outputIndex] = 1.0;
    // mLayers.back()->deactivateDelta(outputIndex);
    //
    // // Apply the delta process recursively for each layer, moving backwards
    // // through the network.
    // for (int i = mLayers.size() - 1; i >= 1; --i)
    // {
    //     mLayers[i]->calculateDeltas(mLayers[i - 1]->getDeltas());
    //     mLayers[i - 1]->deactivateDeltas();
    // }
}

template <class T>
void NeuralNetwork<T>::calculateGradientFromDeltas(const vector<T>& feature, vector<T>& gradient)
{
    const vector<T>* input = &feature;
    size_t weightIndex     = 0;

    for (size_t i = 0; i < mLayers.size(); ++i)
    {
        mLayers[i]->calculateGradient(*input, &gradient[weightIndex]);

        // Get ready for the next iteration
        input = &mLayers[i]->getActivation();
        weightIndex += mLayers[i]->getNumParameters();
    }
}

template <class T>
void NeuralNetwork<T>::calculateGradientParameters(const vector<T>& input, vector<T>& gradient)
{
    const vector<T>* x = &input;
    size_t weightIndex = 0;

    for (size_t i = 0; i < mLayers.size(); ++i)
    {
        mLayers[i]->calculateGradient(*x, &gradient[weightIndex]);

        // Get ready for the next iteration
        x = &mLayers[i]->getActivation();
        weightIndex += mLayers[i]->getNumParameters();
    }
}

template <class T>
void NeuralNetwork<T>::calculateJacobianParameters(const vector<T>& x, Matrix<T>& jacobian)
{
    static vector<T> prediction(getOutputs());
    jacobian.setSize(getOutputs(), mParameters.size());

    // 1. Forward propagation
    evaluate(x, prediction);

    for (size_t i = 0; i < getOutputs(); ++i)
    {
        // 2. Calculate blame terms for all the nodes in the network
        calculateDeltas(i);

        // 3. Relate blame terms to the gradient
        calculateGradientFromDeltas(x, jacobian[i]);
    }
}

template <class T>
void NeuralNetwork<T>::calculateJacobianInputs(const vector<T>& x, Matrix<T>& jacobian)
{
    const size_t N = getInputs();
    const size_t M = getOutputs();

    static vector<T> prediction(M);

    jacobian.setSize(M, N);
    jacobian.setAll(0.0);

    // 1. Forward propagation
    evaluate(x, prediction);

    for (size_t k = 0; k < M; ++k)
    {
        // 2. Calculate blame terms for all the nodes in the network
        calculateDeltas(k);

        // 3. Relate blame terms to the gradient. This operation is the
        // same as backpropagating the deltas in the first layer to the
        // inputs (x).
        mLayers.front()->calculateDeltas(jacobian[k]);
    }
}

template <class T>
void NeuralNetwork<T>::initializeParameters()
{
    std::default_random_engine generator;
    std::normal_distribution<> normal(0.0, 1.0);

    size_t index = 0;
    for (size_t i = 0; i < mLayers.size(); ++i)
    {
        T mag = max(0.03, 1.0 / mLayers[i]->getInputs());
        const size_t N = mLayers[i]->getNumParameters();

        for (size_t j = 0; j < N; ++j)
            mParameters[index++] = normal(generator) * mag;
    }
}

};
#endif /* NEURALNETWORK_H */

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
#include <iostream>
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
    virtual ~NeuralNetwork()
    {
        // Delete the layers that we own
        for (size_t i = 0; i < mLayers.size(); ++i)
        {
            if (mLayerOwnership[i])
            {
                delete mLayers[i];
                mLayers[i] = nullptr;
            }
        }

        // Safeguard
        mLayers.clear();
    };

    // Add a new layer to this Neural Network. Layers are added to the end of
    // the network, so add the layers from input layer to output layer. The
    // caller of this function retains ownership of the layer--it will NOT be
    // destroyed (if necessary) by the network.
    void addLayer(Layer<T>* layer, bool ownLayer = true);

    // Execute one forward pass through the network in order to produce an output.
    void evaluate(const T* input, T* output) override;

    // Execute one forward pass through the network in order to produce an
    // output. As opposed to the first variant of 'evaluate', a set of 'N'
    // inputs are evaluated simultaneously.
    void evaluateBatch(const Matrix<T>& input, Matrix<T>& output);

    // Passes the error from the last layer into each of the nodes preceeding
    // it. This function assumes that the delta values for the last layer have
    // already been calculated manually. (E.g. for SSE, it will be the differences
    // between the target value and the predicted value.)
    void calculateDeltas();

    // Calculates the gradient of the network with respect to the parameters,
    // under the assumption that the deltas have already been calculated for
    // every applicable node in the network.
    void calculateGradientParameters(const T* input, T* gradient);

    // Calculates the gradient of the network with respect to the parameters,
    // under the assumption that the deltas have already been calculated for
    // every applicable node in the network.
    void calculateGradientParametersBatch(const Matrix<T>& input, T* gradient);

    // Calculates the Jacobian of the network with respect to the weights and
    // biases. This involves one forward pass and one backwards pass for each
    // output of the network.
    void calculateJacobianParameters(const T* x, Matrix<T>& jacobian) override;

    // Calculates the Jacobian of the network with respect to the inputs. This
    // involves one forward pass and one backwards pass for each output of the
    // network.
    void calculateJacobianInputs(const T* x, Matrix<T>& jacobian) override;

    // Initializes the weights and biases with random values
    void initializeParameters(Rand& rand);

    // Prints a table of information about this network to the given stream
    void print(std::ostream& out, const string& prefix = "") const;

    // Getters / Setters
    size_t getInputs() const override
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
    vector<bool> mLayerOwnership;
};

template <class T>
void NeuralNetwork<T>::addLayer(Layer<T>* layer, bool ownLayer)
{
    // Make sure this layer is compatible with the rest of the network
    if (!mLayers.empty() && mLayers.back()->getOutputs() != layer->getInputs())
    {
        std::cerr << "This number of inputs to this layer ("
            << layer->getInputs() << ") must match the number of outputs in the"
            << "layer before (" << mLayers.back()->getOutputs() << ")" << std::endl;
        throw Ex("Unable to add layer.");
    }

    // Increase the network's storage to accommodate the new layer. Give the
    // new layer a share of the parameters to work with.
    mParameters.resize(mParameters.size() + layer->getNumParameters());
    mLayers.push_back(layer);
    mLayerOwnership.push_back(ownLayer);

    // When the parameters vector is resized, it's possible that the pointers
    // may have been invalidated. To guard against that, we need to reassign
    // the storage for all of the layers. (Yes, this is inefficient, but it is
    // assumed that the user isn't going to be adding more layers to the network
    // at runtime.)
    T* data = mParameters.data();
    for (Layer<T>*& l : mLayers)
    {
        l->assignStorage(data);
        data += l->getNumParameters();
    }
}

template <class T>
void NeuralNetwork<T>::evaluate(const T* input, T* output)
{
    // Create a matrix that wraps the contents of 'input'.
    Matrix<T> temp((T*) input, 1, getInputs());
    Matrix<T> out(output, 1, getOutputs());

    const Matrix<T>* x = &temp;

    // Feed the output of the previous layer as input
    // to the next layer.
    for (int i = 0; i < mLayers.size() - 1; ++i)
    {
        mLayers[i]->setEffectiveBatchSize(1);
        mLayers[i]->eval(*x);
        x = &mLayers[i]->getActivation();
    }

    // Feed the output of the last layer into 'output'
    mLayers.back()->setEffectiveBatchSize(1);
    mLayers.back()->eval(*x, out);
}

template <class T>
void NeuralNetwork<T>::evaluateBatch(const Matrix<T>& input, Matrix<T>& output)
{
    const size_t N     = input.getRows();
    const Matrix<T>* x = &input;

    // Feed the output of the previous layer as input
    // to the next layer.
    for (int i = 0; i < mLayers.size() - 1; ++i)
    {
        mLayers[i]->setEffectiveBatchSize(N);
        mLayers[i]->eval(*x);
        x = &mLayers[i]->getActivation();
    }

    // Feed the output of the last layer into 'output'
    mLayers.back()->setEffectiveBatchSize(N);
    mLayers.back()->eval(*x, output);
}

template <class T>
void NeuralNetwork<T>::calculateDeltas()
{
    const size_t N = mLayers.front()->getActivation().getRows();

    for (size_t i = mLayers.size() - 1; i >= 1; --i)
    {
        Layer<T>*& current = mLayers[i];
        Layer<T>*& prev    = mLayers[i - 1];

        current->setEffectiveBatchSize(N);
        current->calculateDeltas(prev->getActivation(), prev->getDeltas().data());
    }
}

template <class T>
void NeuralNetwork<T>::calculateGradientParameters(const T* input, T* gradient)
{
    // Create a matrix that wraps the contents of 'input'.
    Matrix<T> temp((T*) input, 1, getInputs());
    const Matrix<T>* x = &temp;

    for (Layer<T>*& l : mLayers)
    {
        l->setEffectiveBatchSize(1);
        l->calculateGradient(*x, gradient);

        // Get ready for the next iteration
        x         = &l->getActivation();
        gradient += l->getNumParameters();
    }
}

template <class T>
void NeuralNetwork<T>::calculateGradientParametersBatch(const Matrix<T>& input, T* gradient)
{
    const size_t N     = input.getRows();
    const Matrix<T>* x = &input;

    for (Layer<T>*& l : mLayers)
    {
        l->setEffectiveBatchSize(N);
        l->calculateGradient(*x, gradient);

        // Get ready for the next iteration
        x         = &l->getActivation();
        gradient += l->getNumParameters();
    }
}

template <class T>
void NeuralNetwork<T>::calculateJacobianParameters(const T* x, Matrix<T>& jacobian)
{
    const size_t N = mParameters.size();
    const size_t M = getOutputs();
    static vector<T> prediction(M);
    jacobian.resize(M, N);
    jacobian.fill(T{});

    // 1. Forward propagation
    evaluate(x, prediction.data());

    for (size_t i = 0; i < M; ++i)
    {
        // Calculate the deltas on the last layer first
        Matrix<T>& outputDeltas = mLayers.back()->getDeltas();
        outputDeltas.fill(T{});
        outputDeltas(0, i) = 1.0;

        // 2. Calculate delta terms for all the other nodes in the network
        calculateDeltas();

        // 3. Relate blame terms to the gradient
        calculateGradientParameters(x, jacobian(i));
    }
}

template <class T>
void NeuralNetwork<T>::calculateJacobianInputs(const T* x, Matrix<T>& jacobian)
{
    const size_t N = getInputs();
    const size_t M = getOutputs();
    static vector<T> prediction(M);
    Matrix<T> input((T*) x, 1, N);
    jacobian.resize(M, N);
    jacobian.fill(T{});

    // 1. Forward propagation
    evaluate(x, prediction.data());

    for (size_t i = 0; i < M; ++i)
    {
        // Calculate the deltas on the last layer first
        Matrix<T>& outputDeltas = mLayers.back()->getDeltas();
        outputDeltas.fill(T{});
        outputDeltas(0, i) = 1.0;

        // 2. Calculate delta terms for all the other nodes in the network
        calculateDeltas();

        // 3. Relate blame terms to the gradient. This operation is the
        // same as backpropagating the deltas in the first layer to the
        // inputs (x).
        mLayers.front()->setEffectiveBatchSize(1);
        mLayers.front()->calculateDeltas(input, jacobian(i));
    }
}

template <class T>
void NeuralNetwork<T>::initializeParameters(Rand& rand)
{
    size_t index = 0;
    for (Layer<T>*& l : mLayers)
    {
        T mag = max(0.03, 1.0 / l->getInputs());
        const size_t N = l->getNumParameters();

        for (size_t j = 0; j < N; ++j)
            mParameters[index++] = rand.nextGaussian(0.0, 1.0) * mag;
    }
}

template <class T>
void NeuralNetwork<T>::print(std::ostream& out, const string& prefix) const
{
    const size_t LINE_LENGTH = 64;
    const char* HEADER_STRING = "%s   %4s | %-22s | %-7s | %-7s | %-10s | %s\n";
    const char* DATA_STRING   = "%s | %4zu | %-22s | %7zu | %7zu | %10zu | %s\n";

    const size_t BUFFER_SIZE = 1024;
    char buffer[BUFFER_SIZE];

    snprintf(buffer, BUFFER_SIZE, HEADER_STRING,
        prefix.c_str(), "", "Type", "Inputs", "Outputs", "Weights", "Misc");
    out << buffer;

    // Print dividing line
    out << prefix << " +";
    for (size_t i = 0; i < LINE_LENGTH; ++i)
        out << "-";
    out << "+\n";

    for (size_t i = 0; i < mLayers.size(); ++i)
    {
        snprintf(buffer, BUFFER_SIZE, DATA_STRING,
            prefix.c_str(),
            i,
            mLayers[i]->getName().c_str(),
            mLayers[i]->getInputs(),
            mLayers[i]->getOutputs(),
            mLayers[i]->getNumParameters(),
            mLayers[i]->getMiscString().c_str());
        out << buffer;
    }

    // Print dividing line
    out << prefix << " +";
    for (size_t i = 0; i < LINE_LENGTH; ++i)
        out << "-";
    out << "+\n";

    // Print summary line
    snprintf(buffer, BUFFER_SIZE, DATA_STRING,
        prefix.c_str(),
        mLayers.size(),
        "Summary",
        mLayers[0]->getInputs(),
        mLayers.back()->getOutputs(),
        getNumParameters(),
        "");
    out << buffer;

    // Print dividing line
    out << prefix << " +";
    for (size_t i = 0; i < LINE_LENGTH; ++i)
        out << "-";
    out << "+\n";
}

};
#endif /* NEURALNETWORK_H */

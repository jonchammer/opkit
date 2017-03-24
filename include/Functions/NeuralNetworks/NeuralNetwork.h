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
// (ANN). A Neural Network consists of layers. The output from one layer is fed
// as input to the next. Layers represent arbitrary functions (usually for which
// derivative information is readily available).
template <class T>
class NeuralNetwork : public Function<T>
{
public:

    // -----------------------------------------------------------------------//
    // Constructors / Destructors
    // -----------------------------------------------------------------------//

    // Create an empty Neural Network. Layers need to be added by calling
    // 'addLayer' before anything useful can be done with it. 'maxBatchSize'
    // determines the largest batch size the network can be expected to handle.
    NeuralNetwork(size_t maxBatchSize = 1) :
        mMaxBatchSize(maxBatchSize)
    {}

    // Destroy the Neural Network
    virtual ~NeuralNetwork();

    // -----------------------------------------------------------------------//
    // Simple Sample Training Methods
    // -----------------------------------------------------------------------//

    // Pass a single sample through the network to produce an output. The size
    // of 'input' must match the number of inputs to the first layer. Similarly,
    // the size of 'output' must be the same as the number of outputs from the
    // layers layer.
    //
    // *Required by the Function interface.
    void evaluate(const T* input, T* output) override;

    // Passes the error from layer 'layerStart' into each preceeding layer
    // until 'layerEnd' has been reached. This function assumes that the delta
    // values for the last layer have already been calculated manually. (E.g.
    // for SSE, it will be the differences between the target value and the
    // predicted value.)
    //
    // 'layerStart' specifies the first layer to work with. This method iterates
    // from the output layer back to the input layer, by default, 'layerStart'
    // is set to the index of the last layer. 'layerEnd' specifies the last
    // layer to work with. Note that this must be >= 1, as continuing the
    // backpropagation algorithm past the first layer requires a different
    // interface.
    void backpropInputsSingle(const size_t layerStart, const size_t layerEnd = 1);

    void backpropInputsSingle()
    {
        backpropInputsSingle(mLayers.size() - 1, 1);
    }

    // This function uses the backpropagation algorithm to determine the
    // partial derivative of the network with respect to each of the parameters
    // in all of the layers for a single training sample.
    void backpropParametersSingle(const T* input, T* gradient);

    // Calculates the Jacobian of the network with respect to the weights and
    // biases. This involves one forward pass and one backwards pass for each
    // output of the network.
    //
    // *Required by the Function interface.
    void calculateJacobianParameters(const T* x, Matrix<T>& jacobian) override;

    // Calculates the Jacobian of the network with respect to the inputs. This
    // involves one forward pass and one backwards pass for each output of the
    // network.
    //
    // *Required by the Function interface.
    void calculateJacobianInputs(const T* x, Matrix<T>& jacobian) override;

    // -----------------------------------------------------------------------//
    // Batch Training Methods
    // -----------------------------------------------------------------------//

    // Pass a batch of 'N' samples through the network to produce 'N' separate
    // outputs. 'inputs' must be an 'N x inputs' matrix, where each row is a
    // single training sample. 'output' must be an 'N x outputs' matrix, where
    // each row will hold one unique output vector.
    void evaluateBatch(const Matrix<T>& input, Matrix<T>& output);

    // Passes the error from layer 'layerStart' into each preceeding layer
    // until 'layerEnd' has been reached. This function operates similarly to
    // 'backpropInputsSingle', except that it works with a batch of training
    // samples, rather than a single sample. As with 'backpropInputsSingle',
    // it is assumed that the deltas for the last layer have already been
    // calculated manually.
    //
    // 'layerStart' specifies the first layer to work with. This method iterates
    // from the output layer back to the input layer, by default, 'layerStart'
    // is set to the index of the last layer. 'layerEnd' specifies the last
    // layer to work with. Note that this must be >= 1, as continuing the
    // backpropagation algorithm past the first layer requires a different
    // interface.
    void backpropInputsBatch(const size_t layerStart, const size_t layerEnd = 1);

    void backpropInputsBatch()
    {
        backpropInputsBatch(mLayers.size() - 1, 1);
    }

    // This function uses the backpropagation algorithm to determine the
    // AVERAGE partial derivative of the network with respect to each of the
    // parameters in all of the layers for a batch of training samples. This
    // function will return a single vector, not a matrix containing the
    // gradient for each sample. To accomplish that, you should call
    // 'backpropParametersSingle' for each sample.
    void backpropParametersBatch(const Matrix<T>& input, T* gradient);

    // -----------------------------------------------------------------------//
    // Layer Manipulation Methods
    // -----------------------------------------------------------------------//

    // Add a new layer to this Neural Network. Layers are added to the end of
    // the network, so add the layers from input layer to output layer. By
    // default, the network owns this layer, so it will be destroyed when the
    // network is destroyed.
    void addLayer(Layer<T>* layer, bool ownLayer = true);

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

    // -----------------------------------------------------------------------//
    // Layer Storage Methods
    // -----------------------------------------------------------------------//

    Matrix<T>& getDeltas(const size_t layer)
    {
        return mDeltas[layer];
    }

    const Matrix<T>& getDeltas(const size_t layer) const
    {
        return mDeltas[layer];
    }

    Matrix<T>& getOutputDeltas()
    {
        return mDeltas.back();
    }

    const Matrix<T>& getOutputDeltas() const
    {
        return mDeltas.back();
    }

    Matrix<T>& getActivation(const size_t layer)
    {
        return mActivations[layer];
    }

    const Matrix<T>& getActivation(const size_t layer) const
    {
        return mActivations[layer];
    }

    size_t getMaxBatchSize() const
    {
        return mMaxBatchSize;
    }

    // -----------------------------------------------------------------------//
    // Parameter Access/Modification Methods
    // -----------------------------------------------------------------------//

    // Initializes parameters with random values.
    void initializeParameters(Rand& rand);

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

    // -----------------------------------------------------------------------//
    // General Network Information
    // -----------------------------------------------------------------------//

    // Prints a table of information about this network to the given stream
    void print(std::ostream& out, const string& prefix = "") const;

    size_t getInputs() const override
    {
        return mLayers.front()->getInputs();
    }

    size_t getOutputs() const override
    {
        return mLayers.back()->getOutputs();
    }

private:

    // The layers themselves
    vector<Layer<T>*> mLayers;
    vector<bool> mLayerOwnership;

    // The parameters for the entire network. Each layer is given a share of
    // these, based on the value returned by Layer::getNumParameters().
    vector<T> mParameters;

    // These store the output of each layer and the derivatives of the network
    // respect to those outputs.
    vector<Matrix<T>> mActivations;
    vector<Matrix<T>> mDeltas;
    size_t mMaxBatchSize;
};

template <class T>
NeuralNetwork<T>::~NeuralNetwork()
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

template <class T>
void NeuralNetwork<T>::evaluate(const T* input, T* output)
{
    T* x = (T*) input;
    T* y = nullptr;

    // Feed the output of the previous layer as input to the next layer.
    for (int i = 0; i < mLayers.size(); ++i)
    {
        y = mActivations[i](0);
        mLayers[i]->forwardSingle(x, y);
        x = y;
    }

    // Copy the output of the last layer into 'output'.
    y = mActivations.back()(0);
    vCopy(y, output, mActivations.back().getCols());
}

template <class T>
void NeuralNetwork<T>::evaluateBatch(const Matrix<T>& input, Matrix<T>& output)
{
    Matrix<T>* x = (Matrix<T>*) &input;
    Matrix<T>* y = nullptr;

    // Feed the output of the previous layer as input
    // to the next layer.
    for (int i = 0; i < mLayers.size(); ++i)
    {
        y = &mActivations[i];
        mLayers[i]->forwardBatch(*x, *y);
        x = y;
    }

    // Copy the output of the last layer into 'output'.
    output.copy(mActivations.back());
}

template <class T>
void NeuralNetwork<T>::backpropInputsSingle(
    const size_t layerStart, const size_t layerEnd)
{
    for (size_t i = layerStart; i >= layerEnd; --i)
    {
        const T* x         = mActivations[i - 1](0);
        const T* y         = mActivations[i](0);
        const T* srcDeltas = mDeltas[i](0);
        T* destDeltas      = mDeltas[i - 1](0);
        mLayers[i]->backpropInputsSingle(x, y, srcDeltas, destDeltas);
    }
}

template <class T>
void NeuralNetwork<T>::backpropInputsBatch(
    const size_t layerStart, const size_t layerEnd)
{
    for (size_t i = layerStart; i >= layerEnd; --i)
    {
        const Matrix<T>& x         = mActivations[i - 1];
        const Matrix<T>& y         = mActivations[i];
        const Matrix<T>& srcDeltas = mDeltas[i];
        Matrix<T>& destDeltas      = mDeltas[i - 1];
        mLayers[i]->backpropInputsBatch(x, y, srcDeltas, destDeltas);
    }
}

template <class T>
void NeuralNetwork<T>::backpropParametersSingle(const T* input, T* gradient)
{
    const T* x = input;
    for (size_t i = 0; i < mLayers.size(); ++i)
    {
        mLayers[i]->backpropParametersSingle(x, mDeltas[i](0), gradient);

        x         = mActivations[i](0);
        gradient += mLayers[i]->getNumParameters();
    }
}

template <class T>
void NeuralNetwork<T>::backpropParametersBatch(
    const Matrix<T>& input, T* gradient)
{
    const Matrix<T>* x = &input;
    for (size_t i = 0; i < mLayers.size(); ++i)
    {
        mLayers[i]->backpropParametersBatch(*x, mDeltas[i], gradient);

        x         = &mActivations[i];
        gradient += mLayers[i]->getNumParameters();
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
        mDeltas.back().fill(T{});
        mDeltas.back()(0, i) = T{1.0};

        // 2. Calculate delta terms for all the other nodes in the network
        backpropInputsSingle();

        // 3. Relate blame terms to the gradient
        backpropParametersSingle(x, jacobian(i));
    }
}

template <class T>
void NeuralNetwork<T>::calculateJacobianInputs(const T* x, Matrix<T>& jacobian)
{
    const size_t N = getInputs();
    const size_t M = getOutputs();
    static vector<T> prediction(M);
    jacobian.resize(M, N);
    jacobian.fill(T{});

    // 1. Forward propagation
    evaluate(x, prediction.data());

    for (size_t i = 0; i < M; ++i)
    {
        // Calculate the deltas on the last layer first
        mDeltas.back().fill(T{});
        mDeltas.back()(0, i) = T{1.0};

        // 2. Calculate delta terms for all the other nodes in the network
        backpropInputsSingle();

        // 3. Relate blame terms to the gradient. This operation is the
        // same as backpropagating the deltas in the first layer to the
        // inputs (x).
        mLayers.front()->backpropInputsSingle(x, mActivations[0](0),
            mDeltas[0](0), jacobian(i));
    }
}

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

    // Make sure this layer makes sense.
    else if (layer->getInputs() == 0 || layer->getOutputs() == 0)
    {
        std::cerr << "The number of inputs and outputs to this layer must be"
            << " > 0" << std::endl;
        throw Ex("Unable to add layer.");
    }

    // Increase the network's storage to accommodate the new layer.
    mParameters.resize(mParameters.size() + layer->getNumParameters());
    mLayers.push_back(layer);
    mLayerOwnership.push_back(ownLayer);
    mActivations.emplace_back(mMaxBatchSize, layer->getOutputs());
    mDeltas.emplace_back(mMaxBatchSize, layer->getOutputs());

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
void NeuralNetwork<T>::print(std::ostream& out, const std::string& prefix) const
{
    const size_t LINE_LENGTH  = 72;
    const char* HEADER_STRING = "%s   %4s | %-30s | %-7s | %-7s | %-10s |\n";
    const char* DATA_STRING   = "%s | %4zu | %-30s | %7zu | %7zu | %10zu |\n";
    const char* MISC_ELEMENT_STRING =
        "%s |      |  - %-27s |         |         |            |\n";

    const size_t BUFFER_SIZE = 1024;
    char buffer[BUFFER_SIZE];

    snprintf(buffer, BUFFER_SIZE, HEADER_STRING,
        prefix.c_str(), "", "Type / Properties", "Inputs", "Outputs", "Weights");
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
            mLayers[i]->getNumParameters());
        out << buffer;

        // Print out the extra information
        size_t numElements;
        string* miscArray = mLayers[i]->getProperties(numElements);

        for (size_t i = 0; i < numElements; ++i)
        {
            snprintf(buffer, BUFFER_SIZE, MISC_ELEMENT_STRING,
                prefix.c_str(), miscArray[i].c_str());
            out << buffer;
        }

        if (miscArray != nullptr)
            delete[] miscArray;
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

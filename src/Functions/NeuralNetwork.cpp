#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork() 
{
    // Do nothing
}

void NeuralNetwork::addLayer(Layer* layer)
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
    layer->assignStorage(&mParameters, origSize);
    
    mLayers.push_back(layer);
}

void NeuralNetwork::evaluate(const vector<double>& input, vector<double>& output)
{
    // For single layer networks, we feed directly into the output
    if (mLayers.size() == 1)
        mLayers[0]->feed(input, output);
    
    else
    {
        // Feed the input to the first layer and put the result in Layer 1's 
        // activation
        mLayers[0]->feed(input);

        // Feed the activation from the previous layer into the current layer.
        for (size_t i = 1; i < mLayers.size() - 1; ++i)
            mLayers[i]->feed(mLayers[i - 1]->getActivation());

        // On the last layer, feed the previous layer's activation and put the
        // result in 'output'
        mLayers.back()->feed(mLayers[mLayers.size() - 2]->getActivation(), output);
    }
}

void NeuralNetwork::calculateDeltas(const size_t outputIndex)
{
    // Calculate the deltas on the last layer first
    vector<double>& outputDeltas = mLayers.back()->getDeltas();
    std::fill(outputDeltas.begin(), outputDeltas.end(), 0.0);
    outputDeltas[outputIndex] = 1.0;
    mLayers.back()->deactivateDelta(outputIndex);
    
    // Apply the delta process recursively for each layer, moving backwards
    // through the network.
    for (int i = mLayers.size() - 1; i >= 1; --i)
    {
        mLayers[i]->calculateDeltas(mLayers[i - 1]->getDeltas());
        mLayers[i - 1]->deactivateDeltas();
    }
}

void NeuralNetwork::calculateGradientFromDeltas(const vector<double>& feature,
    vector<double>& gradient)
{
    const vector<double>* input = &feature;
    size_t weightIndex          = 0;
    
    for (size_t i = 0; i < mLayers.size(); ++i)
    {
        mLayers[i]->calculateGradient(*input, &gradient[weightIndex]);
        
        // Get ready for the next iteration
        input = &mLayers[i]->getActivation();
        weightIndex += mLayers[i]->getNumParameters();
    }
}

void NeuralNetwork::calculateJacobianParameters(const vector<double>& x, 
    Matrix& jacobian)
{
    static vector<double> prediction;
    prediction.resize(getOutputs());
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

void NeuralNetwork::calculateJacobianInputs(const vector<double>& x, 
        Matrix& jacobian)
{
    const size_t N = getInputs();
    const size_t M = getOutputs();
    
    static vector<double> prediction;
    prediction.resize(M);
    
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

size_t NeuralNetwork::getInputs()  const
{
    return mLayers.front()->getInputs();
}

size_t NeuralNetwork::getOutputs() const
{
    return mLayers.back()->getOutputs();
}

vector<double>& NeuralNetwork::getParameters()
{
    return mParameters;
}

const vector<double>& NeuralNetwork::getParameters() const
{
    return mParameters;
}

size_t NeuralNetwork::getNumParameters() const
{
    return mParameters.size();
}
 
bool NeuralNetwork::cachesLastEvaluation() const
{
    return true;
}

void NeuralNetwork::getLastEvaluation(vector<double>& output)
{
    vector<double>& lastActivation = mLayers.back()->getActivation();
    std::copy(lastActivation.begin(), lastActivation.end(), output.begin());
}

size_t NeuralNetwork::getNumLayers() const
{
    return mLayers.size();
}

Layer* NeuralNetwork::getLayer(const size_t index)
{
    return mLayers[index];
}

const Layer* NeuralNetwork::getLayer(const size_t index) const
{
    return mLayers[index];
}

Layer* NeuralNetwork::getOutputLayer()
{
    return mLayers.back();
}

const Layer* NeuralNetwork::getOutputLayer() const
{
    return mLayers.back();
}

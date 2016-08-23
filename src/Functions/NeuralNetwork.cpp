#include "NeuralNetwork.h"

// Layer implementations
Layer::Layer(size_t inputs, size_t outputs, 
    vector<double>& parameterStorage, int parameterStartIndex)
    : mParameters(parameterStorage), mParameterStartIndex(parameterStartIndex), 
      mInputs(inputs), mOutputs(outputs)
{
    mBlame.resize(outputs);
    mNet.resize(outputs);
    mActivation.resize(outputs);
    
    // Use tanh() as the default activation function
    mActFunction = tanhActivation;
}
    
void Layer::feed(const vector<double>& x, vector<double>& y)
{
    y.resize(mOutputs, 0.0);
    
    // Cache the beginning of the bias section of the parameters
    size_t biasStart = mInputs * mOutputs;
    
    // The weights are arranged so they can be examined in order
    size_t weightIndex = 0;
    
    for (size_t j = 0; j < mOutputs; ++j)
    {
        // sum = W * x
        mNet[j] = 0.0;

        for (size_t i = 0; i < mInputs; ++i)
        {
            mNet[j] += mParameters[mParameterStartIndex + weightIndex] * x[i];
            weightIndex++;
        }
        mNet[j] += mParameters[mParameterStartIndex + biasStart + j];
        
        // y = a(sum + bias)
        y[j] = (*mActFunction.first)(mNet[j]);
        
        // Usually, y will be mActivation, but there's no need for that to be
        // the case. If not, we need to make sure to set mActivation manually,
        // since some algorithms depend on it.
        mActivation[j] = y[j];
    }
}

// Neural Network implementations
NeuralNetwork::NeuralNetwork(const vector<int>& layerDimensions)
    : StandardFunction(layerDimensions.front(), layerDimensions.back(), 0)
{
    // Sanity check
	if (layerDimensions.size() < 2)
		throw Ex("At least 2 dimensions must be provided.");

    // Sort out how many parameters we need
    size_t numParameters = 0;
    for (size_t i = 1; i < layerDimensions.size(); ++i)
    {
        // Account for the weights
        numParameters += layerDimensions[i - 1] * layerDimensions[i];
        
        // Account for the biases
        numParameters += layerDimensions[i];
    }
    mParameters.resize(numParameters);
    
	// Create each of the layers, giving each some of our parameters
    size_t paramIndex = 0;
	for (size_t i = 1; i < layerDimensions.size(); ++i)
	{
		Layer l(layerDimensions[i - 1], layerDimensions[i], 
            mParameters, paramIndex);
		mLayers.push_back(l);
        paramIndex += layerDimensions[i - 1] * layerDimensions[i] +
            layerDimensions[i];
	}
}

void NeuralNetwork::evaluate(const vector<double>& input, 
    vector<double>& output)
{
    // For single layer networks, we feed directly into the output
    if (mLayers.size() == 1)
        mLayers[0].feed(input, output);
    
    else
    {
        // Feed the input to the first layer and put the result in Layer 1's 
        // activation
        mLayers[0].feed(input);

        // Feed the activation from the previous layer into the current layer.
        for (size_t i = 1; i < mLayers.size() - 1; ++i)
            mLayers[i].feed(mLayers[i - 1].getActivation());

        // On the last layer, feed the previous layer's activation and put the
        // result in 'output'
        mLayers.back().feed(mLayers[mLayers.size() - 2].getActivation(), output);
    }
}

void NeuralNetwork::calculateBlameTerms(const size_t outputIndex)
{
    // Calculate the blame terms for the output nodes
    Layer* current        = &mLayers.back();
    vector<double>* blame = &current->getBlame();
    vector<double>* net   = &current->getNet();
    vector<double>* act   = &current->getActivation();

    // Blame is set to 0 for all outputs except the one we're interested in
    std::fill(blame->begin(), blame->end(), 0.0);
    (*blame)[outputIndex] = 
        (*current->getActivationFunction().second)  // Get the activation function's derivative
        ((*net)[outputIndex], (*act)[outputIndex]); // Apply it to the net outputs
    
    // Calculate blame terms for the interior nodes
    size_t weightIndex = mParameters.size();

    for (size_t layer = mLayers.size() - 1; layer >= 1; --layer)
    {
        Layer* right = &mLayers[layer];
        Layer* left  = &mLayers[layer - 1];

        vector<double>* rightBlame = &right->getBlame();
        vector<double>* leftBlame  = &left->getBlame();
        vector<double>* leftNet    = &left->getNet();
        vector<double>* leftAct    = &left->getActivation();
        std::fill(leftBlame->begin(), leftBlame->end(), 0.0);

        int rightInputs  = right->getInputSize();
        int rightOutputs = right->getOutputSize();
        weightIndex      -= (rightInputs + 1) * rightOutputs;
        
        for (int i = 0; i < rightInputs; ++i)
        {
            double actDerivative = (*left->getActivationFunction().second)
                ((*leftNet)[i], (*leftAct)[i]);

            for (int j = 0; j < rightOutputs; ++j)
            {
                double weight = mParameters[weightIndex + (j * rightInputs + i)];
                (*leftBlame)[i] += weight * (*rightBlame)[j] * actDerivative;
            }
        }
    }    
}

void NeuralNetwork::calculateGradientFromBlame(const vector<double>& feature, 
    vector<double>& gradient)
{
    size_t weightIndex          = 0;
    size_t layer                = 1;
    Layer* current              = &mLayers.front();
    const vector<double>* input = &feature;

    do
    {
        int inputSize  = current->getInputSize();
        int outputSize = current->getOutputSize();
        
        // Calculate gradient for the weights
        vector<double>& blame = current->getBlame();
        for (int i = 0; i < outputSize; ++i)
        {
            for (int j = 0; j < inputSize; ++j)
                gradient[weightIndex++] = (*input)[j] * blame[i];
        }

        // Calculate gradient for the biases
        for (int i = 0; i < outputSize; ++i)
            gradient[weightIndex++] = blame[i];

        // Get ready for the next layer
        input   = &current->getActivation();
        current = &mLayers[layer];
        ++layer;
    }
    while (layer <= mLayers.size());
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
        calculateBlameTerms(i);

        // 3. Relate blame terms to the gradient
        calculateGradientFromBlame(x, jacobian[i]);
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
        calculateBlameTerms(k);

        // 3. Relate blame terms to the gradient
        vector<double>& blame = mLayers.front().getBlame();
        int weightIndex       = 0;

        for (size_t j = 0; j < blame.size(); ++j)
        {
            for (size_t i = 0; i < N; ++i)
                jacobian[k][i] += mParameters[weightIndex++] * blame[j];
        }
    }
}

size_t NeuralNetwork::getNumLayers() const
{
    return mLayers.size();
}

Layer& NeuralNetwork::getLayer(const size_t index)
{
    return mLayers[index];
}

const Layer& NeuralNetwork::getLayer(const size_t index) const
{
    return mLayers[index];
}

Layer& NeuralNetwork::getOutputLayer()
{
    return mLayers.back();
}

const Layer& NeuralNetwork::getOutputLayer() const
{
    return mLayers.back();
}
    
//------------------OLD------------------
//void NeuralNetwork::calculateBlameTerms(const vector<double>& label, 
//    const vector<double>& prediction)
//{
//    // Calculate the blame terms for the output nodes
//    Layer* current        = &mLayers.back();
//    vector<double>* blame = &current->getBlame();
//    vector<double>* net   = &current->getNet();
//
//    // d/dx[tanh(x)] = 1 - tanh(x)^2
//    for (size_t i = 0; i < label.size(); ++i)
//        (*blame)[i] = (label[i] - prediction[i]) * (1.0 - (*net)[i] * (*net)[i]);
//    
//    // Calculate blame terms for the interior nodes
//    size_t weightIndex = mParameters.size() - 
//        (mLayers.back().getInputSize() + 1) * mLayers.back().getOutputSize();
//
//    for (size_t layer = mLayers.size() - 1; layer >= 1; --layer)
//    {
//        Layer* right = &mLayers[layer];
//        Layer* left  = &mLayers[layer - 1];
//
//        vector<double>* rightBlame = &right->getBlame();
//        vector<double>* leftBlame  = &left->getBlame();
//        vector<double>* leftNet    = &left->getNet();
//        std::fill(leftBlame->begin(), leftBlame->end(), 0.0);
//
//        int rightInputs  = right->getInputSize();
//        int rightOutputs = right->getOutputSize();
//        
//        for (int i = 0; i < rightInputs; ++i)
//        {
//            double actDerivative = 1.0 - (*leftNet)[i] * (*leftNet)[i];
//
//            for (int j = 0; j < rightOutputs; ++j)
//            {
//                double weight = mParameters[weightIndex + (j * rightInputs + i)];
//                (*leftBlame)[i] += weight * (*rightBlame)[j] * actDerivative;
//            }
//        }
//    }    
//}
//
//void NeuralNetwork::calculateGradientFromBlame(const vector<double>& feature, 
//    vector<double>& gradient)
//{
//    size_t weightIndex          = 0;
//    size_t layer                = 1;
//    Layer* current              = &mLayers.front();
//    const vector<double>* input = &feature;
//
//    do
//    {
//        int inputSize  = current->getInputSize();
//        int outputSize = current->getOutputSize();
//        
//        // Calculate gradient for the weights
//        vector<double>& blame = current->getBlame();
//        for (int i = 0; i < outputSize; ++i)
//        {
//            for (int j = 0; j < inputSize; ++j)
//                gradient[weightIndex++] = -2 * (*input)[j] * blame[i];
//        }
//
//        // Calculate gradient for the biases
//        for (int i = 0; i < outputSize; ++i)
//            gradient[weightIndex++] = -2 * blame[i];
//
//        // Get ready for the next layer
//        input   = &current->getActivation();
//        current = &mLayers[layer];
//        ++layer;
//    }
//    while (layer <= mLayers.size());
//}

//void NeuralNetwork::calculateGradient(const vector<double>& feature, 
//        const vector<double>& label, vector<double>& gradient)
//{
//    static vector<double> prediction;
//    prediction.resize(label.size());
//    gradient.resize(mParameters.size());
//    
//    // 1. Forward propagation
//    evaluate(feature, prediction);
//        
//    // 2. Calculate blame terms for the output nodes
//    calculateBlameTerms(label, prediction);
//    
//    // 3. Relate blame terms to the gradient
//    calculateGradientFromBlame(feature, gradient);
//}
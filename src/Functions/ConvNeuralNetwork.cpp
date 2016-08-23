#include "ConvNeuralNetwork.h"

// ---- Layer Implementations ---- //
void Layer::feed(const vector<double>& x)
{
    feed(x, getActivation());
}

void Layer::assignStorage(vector<double>* parameters, 
    const size_t parametersStartIndex)
{
    mParameters           = parameters;
    mParametersStartIndex = parametersStartIndex;
}

// ---- FeedforwardLayer Implementations ---- //

FeedforwardLayer::FeedforwardLayer(size_t inputs, size_t outputs)
    : mInputs(inputs), mOutputs(outputs)
{
    mBlame.resize(outputs);
    mNet.resize(outputs);
    mActivation.resize(outputs);
    
    // Use tanh() as the default activation function
    mActFunction = tanhActivation;
}
    
void FeedforwardLayer::feed(const vector<double>& x, vector<double>& y)
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
            mNet[j] += (*mParameters)[mParametersStartIndex + weightIndex] * x[i];
            weightIndex++;
        }
        mNet[j] += (*mParameters)[mParametersStartIndex + biasStart + j];
        
        // y = a(sum + bias)
        y[j] = (*mActFunction.first)(mNet[j]);
        
        // Usually, y will be mActivation, but there's no need for that to be
        // the case. If not, we need to make sure to set mActivation manually,
        // since some algorithms depend on it.
        mActivation[j] = y[j];
    }
}

size_t FeedforwardLayer::getNumParameters()       
{ 
    return (mInputs + 1) * mOutputs; 
}

size_t FeedforwardLayer::getInputs()               
{ 
    return mInputs;                  
}

size_t FeedforwardLayer::getOutputs()              
{ 
    return mOutputs;                 
}

vector<double>& FeedforwardLayer::getActivation()  
{ 
    return mActivation;              
}
   
vector<double>& FeedforwardLayer::getBlame()      
{ 
    return mBlame;      
}

vector<double>& FeedforwardLayer::getNet()        
{ 
    return mNet;        
}

Activation FeedforwardLayer::getActivationFunction()         
{ 
    return mActFunction; 
}

void FeedforwardLayer::setActivationFunction(Activation act) 
{ 
    mActFunction = act;  
}

// ---- ConvLayer Implementations ---- //

ConvLayer::ConvLayer(size_t inputWidth, size_t inputHeight, size_t inputChannels, 
    size_t filterSize, size_t numFilters, size_t stride, size_t zeroPadding)
    : mInputWidth(inputWidth), mInputHeight(inputHeight), 
    mInputChannels(inputChannels), mFilterSize(filterSize), 
    mNumFilters(numFilters), mStride(stride), mZeroPadding(zeroPadding)
{
    // Check to make sure the given metaparameters are actually a valid
    // configuration. We have to make sure that this equation:
    // (inputWidth - filterSize + (2 * zeroPadding)) / stride
    // produces a whole number.
    size_t num = inputWidth - filterSize + 2 * zeroPadding;
    if (num % stride != 0)
    {
        cerr << "Error: Stride = " << stride << " does not evenly divide " 
            << num << endl;
        throw Ex("Unable to create convolutional layer.");
    }
    
    // Ensure the vectors are big enough to hold the data
    size_t outputSize = getOutputWidth() * getOutputHeight() * mNumFilters;
    mNet.resize(outputSize);
    mActivation.resize(outputSize);
    
    // Use tanh() as the default activation function
    mActFunction = tanhActivation;
}

// (x, y, z) specifies coordinates of the output cell we are working on
double ConvLayer::convolve(Tensor3D& input, size_t x, size_t y, size_t z)
{
    // Calculate where the weights and bias values are for this filter
    size_t numFilterWeights = mFilterSize * mFilterSize * mInputChannels;
    size_t weightsIndex     = z * (numFilterWeights + 1);
    size_t biasIndex        = weightsIndex + numFilterWeights;
    Tensor3D filterWeights(*mParameters, weightsIndex, 
        mFilterSize, mFilterSize, mInputChannels);
    
    // Calculate the bounds of the input window
    // The amount of padding dictates where the top left corner will start. To
    // that, we add the product of the index and the stride to move the box
    // either horizontally or vertically. The right and bottom bounds are found
    // by adding the filter size to the left/top bounds.
    int lowX = -mZeroPadding + x * mStride;
    int lowY = -mZeroPadding + y * mStride;
    int hiX  = lowX + mFilterSize - 1;
    int hiY  = lowY + mFilterSize - 1;
    
    // Do the actual convolution. Tensor3D objects will return 0 when the 
    // provided index is out of bounds, which is used to implement the zero
    // padding.
    double sum = 0.0;
    for (size_t k = 0; k < mInputChannels; ++k)
    {
        for (int j = lowY; j <= hiY; ++j)
        {
            for (int i = lowX; i <= hiX; ++i)
                sum += input.get(i, j, k) * 
                    filterWeights.get(i - lowX, j - lowY, k);
        }
    }
    
    // Factor in the bias
    return sum + (*mParameters)[biasIndex];
}

void ConvLayer::feed(const vector<double>& x, vector<double>& y)
{
    Tensor3D input((vector<double>&) x, 0, mInputWidth, mInputHeight, mInputChannels);
    
    size_t outputWidth  = getOutputWidth();
    size_t outputHeight = getOutputHeight();
    size_t outputDepth  = mNumFilters;
    y.resize(outputWidth * outputHeight * outputDepth);
    
    Tensor3D net(mNet, 0, outputWidth, outputHeight, outputDepth);
    Tensor3D activation(mActivation, 0, outputWidth, outputHeight, outputDepth);
    Tensor3D output(y, 0, outputWidth, outputHeight, outputDepth);
    
    // Note: These loops can be run in any order. Each iteration is completely
    // independent of every other iteration.
    for (size_t filter = 0; filter < mNumFilters; ++filter)
    {
        for (size_t i = 0; i < outputHeight; ++i)
        {
            for (size_t j = 0; j < outputWidth; ++j)
            {
                // Dot product input about (j, i) with filter 'filter'
                double sum = convolve(input, j, i, filter);
                double act = (*mActFunction.first)(sum);
                
                net.set(j, i, filter, sum);
                activation.set(j, i, filter, act);
                output.set(j, i, filter, act);
            }
        }
    }
}
    
size_t ConvLayer::getNumParameters()
{
    size_t filterParams = mFilterSize * mFilterSize * mInputChannels + 1;
    return filterParams * mNumFilters;
}

size_t ConvLayer::getInputs()
{
    return mInputWidth * mInputHeight * mInputChannels;
}

size_t ConvLayer::getOutputs()
{
    size_t w = (mInputWidth - mFilterSize + 2 * mZeroPadding) / mStride + 1;
    size_t h = (mInputHeight - mFilterSize + 2 * mZeroPadding) / mStride + 1;
    size_t d = mNumFilters;
    
    return w * h * d;
}

vector<double>& ConvLayer::getActivation()
{
    return mActivation;
}

vector<double>& ConvLayer::getNet()
{
    return mNet;
}

size_t ConvLayer::getOutputWidth()
{
    return (mInputWidth - mFilterSize + 2 * mZeroPadding) / mStride + 1;
}

size_t ConvLayer::getOutputHeight()
{
    return (mInputHeight - mFilterSize + 2 * mZeroPadding) / mStride + 1;
}

Activation ConvLayer::getActivationFunction()         
{ 
    return mActFunction; 
}

void ConvLayer::setActivationFunction(Activation act) 
{ 
    mActFunction = act;  
}
    
// ---- NeuralNetwork Implementations ---- //

NeuralNetwork::NeuralNetwork() {}

void NeuralNetwork::addLayer(Layer* layer)
{
    // Increase the network's storage to accommodate the new layer. Give the
    // new layer a share of the parameters to work with.
    size_t numParams = layer->getNumParameters();
    size_t origSize  = mParameters.size();
    mParameters.resize(origSize + numParams);
    layer->assignStorage(&mParameters, origSize);
    
    mLayers.push_back(layer);
}

void NeuralNetwork::evaluate(const vector<double>& input, 
    vector<double>& output)
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

/*
void NeuralNetwork::calculateBlameTerms(const size_t outputIndex)
{
    // Calculate the blame terms for the output nodes
    ConvLayer* current        = &mLayers.back();
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
        ConvLayer* right = &mLayers[layer];
        ConvLayer* left  = &mLayers[layer - 1];

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
    ConvLayer* current              = &mLayers.front();
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
*/

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
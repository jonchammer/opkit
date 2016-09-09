#include "Layer.h"

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
    mDeltas.resize(outputs);
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

void FeedforwardLayer::calculateDeltas(Layer* downstream)
{
    vector<double>& downstreamDeltas = downstream->getDeltas();
    std::fill(downstreamDeltas.begin(), downstreamDeltas.end(), 0.0);

    Activation& act = downstream->getActivationFunction();
    
    for (size_t i = 0; i < mInputs; ++i)
    {
        double actDerivative = (*act.second)
            (downstream->getNet()[i], downstream->getActivation()[i]);

        for (size_t j = 0; j < mOutputs; ++j)
        {
            size_t index  = mParametersStartIndex + (j * mInputs + i);
            double weight = (*mParameters)[index];
            downstreamDeltas[i] += weight * mDeltas[j] * actDerivative;
        }
    }
}

void FeedforwardLayer::calculateDeltas(size_t outputIndex)
{
    // Blame is set to 0 for all outputs except the one we're interested in
    std::fill(mDeltas.begin(), mDeltas.end(), 0.0);
    
    // Apply the derivative of the activation function to the output of this
    // layer
    mDeltas[outputIndex] = (*mActFunction.second)
        (mNet[outputIndex], mActivation[outputIndex]);
}

void FeedforwardLayer::calculateGradient(const vector<double>& input, double* gradient)
{
    int inputSize  = getInputs();
    int outputSize = getOutputs();

    // Calculate gradient for the weights
    size_t index = 0;
    for (int i = 0; i < outputSize; ++i)
    {
        for (int j = 0; j < inputSize; ++j)
            gradient[index++] = input[j] * mDeltas[i];
    }

    // Calculate gradient for the biases
    for (int i = 0; i < outputSize; ++i)
        gradient[index++] = mDeltas[i];
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
   
vector<double>& FeedforwardLayer::getDeltas()      
{ 
    return mDeltas;      
}

vector<double>& FeedforwardLayer::getNet()        
{ 
    return mNet;        
}

Activation& FeedforwardLayer::getActivationFunction()         
{ 
    return mActFunction; 
}

void FeedforwardLayer::setActivationFunction(Activation act) 
{ 
    mActFunction = act;  
}

// ---- Convolutional2DLayer Implementations ---- //

Convolutional2DLayer::Convolutional2DLayer(size_t inputWidth, size_t inputHeight, size_t inputChannels, 
    size_t filterSize, size_t numFilters, size_t stride, size_t zeroPadding)
    : mInputWidth(inputWidth), mInputHeight(inputHeight), 
    mInputChannels(inputChannels), mFilterSize(filterSize), 
    mNumFilters(numFilters), mStride(stride), mZeroPadding(zeroPadding),
    mOutputWidth((inputWidth - filterSize + 2 * zeroPadding) / stride + 1),
    mOutputHeight((inputHeight - filterSize + 2 * zeroPadding) / stride + 1)
{
    if (stride < 1)
    {
        cerr << "Error: Stride must be at least 1." << endl;
        throw Ex("Unable to create convolutional layer.");
    }
    
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
    size_t outputSize = mOutputWidth * mOutputHeight * mNumFilters;
    mNet.resize(outputSize);
    mActivation.resize(outputSize);
    mDeltas.resize(outputSize);
    
    // Use tanh() as the default activation function
    mActFunction = tanhActivation;
}

// (x, y, z) specifies coordinates of the output cell we are working on
double Convolutional2DLayer::convolve(Tensor3D& input, size_t x, size_t y, size_t z)
{
    // Calculate where the weights and bias values are for this filter
    size_t numFilterWeights = mFilterSize * mFilterSize * mInputChannels;
    size_t weightsIndex     = mParametersStartIndex + z * (numFilterWeights + 1);
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

void Convolutional2DLayer::feed(const vector<double>& x, vector<double>& y)
{
    Tensor3D input((vector<double>&) x, 0, mInputWidth, mInputHeight, mInputChannels);
    
    // Make sure the output is the correct size
    y.resize(mOutputWidth * mOutputHeight * mNumFilters);
    
    // Wrap the important vectors in Tensor3D objects so we can work with them
    Tensor3D net(mNet, 0, mOutputWidth, mOutputHeight, mNumFilters);
    Tensor3D activation(mActivation, 0, mOutputWidth, mOutputHeight, mNumFilters);
    Tensor3D output(y, 0, mOutputWidth, mOutputHeight, mNumFilters);
    
    // Note: These loops can be run in any order. Each iteration is completely
    // independent of every other iteration.
    for (size_t filter = 0; filter < mNumFilters; ++filter)
    {
        for (size_t i = 0; i < mOutputHeight; ++i)
        {
            for (size_t j = 0; j < mOutputWidth; ++j)
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
    
void Convolutional2DLayer::calculateDeltas(Layer* downstream)
{
    vector<double>& downstreamDeltas = downstream->getDeltas();
    std::fill(downstreamDeltas.begin(), downstreamDeltas.end(), 0.0);
    
    size_t numFilterWeights = mFilterSize * mFilterSize * mInputChannels + 1;
    
    // Wrap the important vectors in Tensor3D objects so access is easier
    Tensor3D currentDeltas(mDeltas, 0, mOutputWidth, mOutputHeight, mNumFilters);
    Tensor3D targetDeltas(downstreamDeltas, 0, mInputWidth, mInputHeight, mInputChannels);
    
    // Convolve the filters with the deltas for this layer to get the
    // deltas for the next layer downstream (to the left)
    for (size_t k = 0; k < mNumFilters; ++k)
    {
        // Wrap the current filter in a Tensor3D object
        Tensor3D currentFilter(*mParameters, mParametersStartIndex + (k * numFilterWeights), 
            mFilterSize, mFilterSize, mInputChannels);
        
        for (size_t j = 0; j < mOutputHeight; ++j)
        {
            int inRowOffset = j * mStride - mZeroPadding;
            for (size_t i = 0; i < mOutputWidth; ++i)
            {
                int inColOffset = i * mStride - mZeroPadding;
                for (size_t z = 0; z < mInputChannels; ++z)
                {
                    for (size_t y = 0; y < mFilterSize; ++y)
                    {
                        int inRow = inRowOffset + y;
                        for (size_t x = 0; x < mFilterSize; ++x)
                        {
                            int inCol = inColOffset + x;
                            if (inRow >= 0 && inRow < (int)mInputHeight && inCol >= 0 && inCol < (int) mInputWidth)
                            {
                                double val = currentFilter.get(x, y, z) * currentDeltas.get(i, j, k);
                                targetDeltas.add(inCol, inRow, z, val);
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Multiply targetDeltas by the derivative of activation function to 
    // finish deactivating them.
    Activation& act = downstream->getActivationFunction();
    for (size_t i = 0; i < mOutputWidth * mOutputHeight * mNumFilters; ++i)
    {
        double actDerivative = (*act.second)
            (downstream->getNet()[i], downstream->getActivation()[i]);
        downstreamDeltas[i] *= actDerivative;
    }
}

void Convolutional2DLayer::calculateDeltas(size_t outputIndex)
{
    // Blame is set to 0 for all outputs except the one we're interested in
    std::fill(mDeltas.begin(), mDeltas.end(), 0.0);
    
    // Apply the derivative of the activation function to the output of this layer
    mDeltas[outputIndex] = (*mActFunction.second)
        (mNet[outputIndex], mActivation[outputIndex]);
}

void Convolutional2DLayer::calculateGradient(const vector<double>& x, double* gradient)
{    
    size_t numFilterWeights = mFilterSize * mFilterSize * mInputChannels + 1;

    // Wrap the important vectors in Tensor3D objects so access is easier
    Tensor3D input((vector<double>&) x, 0, mInputWidth, mInputHeight, mInputChannels);
    Tensor3D deltas(mDeltas, 0, mOutputWidth, mOutputHeight, mNumFilters);
    
    // Convolve the deltas in this layer with the input in order to calculate
    // the gradient for the weights in the filters. We also calculate the gradient
    // with respect to the bias terms along the way. g(bias) = sum(deltas)
    for (size_t k = 0; k < mNumFilters; ++k)
    {
        Tensor3D grad(gradient, k * numFilterWeights, mFilterSize, mFilterSize, mInputChannels);
        
        size_t outChannelOffset = k * mOutputHeight * mOutputWidth;
        for (size_t j = 0; j < mOutputHeight; ++j)
        {
            size_t outRowOffset = j * mOutputWidth;
            int inRowOffset     = j * mStride - mZeroPadding;
            for (size_t i = 0; i < mOutputWidth; ++i)
            {
                size_t index    = outChannelOffset + outRowOffset + i;
                int inColOffset = i * mStride - mZeroPadding;
                
                // Calculate the gradient of the bias for this filter
                gradient[(k + 1) * numFilterWeights - 1] += mDeltas[index];
                
                // Calculate the gradient of the weights for this filter
                for (size_t z = 0; z < mInputChannels; ++z)
                {
                    for (size_t y = 0; y < mFilterSize; ++y)
                    {
                        int inRow = inRowOffset + y;
                        for (size_t x = 0; x < mFilterSize; ++x)
                        {
                            int inCol = inColOffset + x;
                            if (inRow >= 0 && inRow < (int) mInputHeight && inCol >= 0 && inCol < (int) mInputWidth)
                            {
                                double val = deltas.get(i, j, k) * input.get(inCol, inRow, z);
                                grad.add(x, y, z, val);
                            }
                        }
                    }
                }
            }
        }
    }
}

size_t Convolutional2DLayer::getNumParameters()
{
    size_t filterParams = mFilterSize * mFilterSize * mInputChannels + 1;
    return filterParams * mNumFilters;
}

size_t Convolutional2DLayer::getInputs()
{
    return mInputWidth * mInputHeight * mInputChannels;
}

size_t Convolutional2DLayer::getOutputs()
{
    return mOutputWidth * mOutputHeight * mNumFilters;
}

vector<double>& Convolutional2DLayer::getActivation()
{
    return mActivation;
}

vector<double>& Convolutional2DLayer::getNet()
{
    return mNet;
}

vector<double>& Convolutional2DLayer::getDeltas()
{
    return mDeltas;
}

size_t Convolutional2DLayer::getOutputWidth()
{
    return mOutputWidth;
}

size_t Convolutional2DLayer::getOutputHeight()
{
    return mOutputHeight;
}

size_t Convolutional2DLayer::getInputWidth()
{
    return mInputWidth;
}

size_t Convolutional2DLayer::getInputHeight()
{
    return mInputHeight;
}

size_t Convolutional2DLayer::getInputChannels()
{
    return mInputChannels;
}

size_t Convolutional2DLayer::getFilterSize()
{
    return mFilterSize;
}

size_t Convolutional2DLayer::getNumFilters()
{
    return mNumFilters;
}

size_t Convolutional2DLayer::getStride()
{
    return mStride;
}

size_t Convolutional2DLayer::getZeroPadding()
{
    return mZeroPadding;
}
    
Activation& Convolutional2DLayer::getActivationFunction()         
{ 
    return mActFunction; 
}

void Convolutional2DLayer::setActivationFunction(Activation act) 
{ 
    mActFunction = act;  
}

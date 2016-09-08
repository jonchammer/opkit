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

void FeedforwardLayer::calculateGradient(const vector<double>& input, 
    size_t /*outputIndex*/, double* gradient)
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
    mDeltas.resize(outputSize);
    
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

void ConvLayer::convolve(
    Tensor3D& input,  size_t ix, size_t iy, size_t iz,
    Tensor3D& filter, size_t filterZ,
    Tensor3D& output, size_t outputZ)
{
    // Define the small region inside 'input' that was used during the forward
    // convolution pass
    int lowX = -mZeroPadding + ix * mStride;
    int lowY = -mZeroPadding + iy * mStride;
    int hiX  = lowX + mFilterSize - 1;
    int hiY  = lowY + mFilterSize - 1;
    
    // Create a virtual window around that region that will be padded with
    // 0's on one or more sides.
    int virtualSize = (3 * mFilterSize + 2 * mZeroPadding - 1) / 2;
    int minX = lowX - ix;
    int minY = lowY - iy;
    int maxX = minX + virtualSize - 1;
    int maxY = minY + virtualSize - 1;
    
    for (size_t y = 0; y < output.getHeight(); ++y)
    {
        for (size_t x = 0; x < output.getWidth(); ++x)
        {
            double sum = 0.0;
            
            for (int j = minY; j <= maxY; ++j)
            {
                for (int i = minX; i <= maxX; ++i)
                {
                    // Even if (i + x, j + y) would normally be a valid index,
                    // we still have to make sure we're within the virtual window
                    if (i + (int)x >= lowX && i + (int)x <= hiX && 
                        j + (int)y >= lowY && j + (int)y <= hiY)
                    {
                        sum += input.get(i + (int)x, j + (int)y, iz) * 
                            filter.get(i - minX, j - minY, filterZ);
                    }
                }
            }
            
            output.set(x, y, outputZ, sum);
        }
    }
}

void ConvLayer::convolve2(
    Tensor3D& input,   size_t inputZ, size_t padding, size_t stride,
    Tensor3D& filter,  size_t filterZ,
    Tensor3D& output,  size_t outputZ)
{
    // Go over each cell in this slice of the output volume
    for (size_t y = 0; y < output.getHeight(); ++y)    
    {
        for (size_t x = 0; x < output.getWidth(); ++x)
        {
            // Do the actual convolution for this cell
            double sum = 0.0;
            
            int lowX = -padding + x * stride;
            int lowY = -padding + y * stride;
            int hiX  = lowX + filter.getWidth();
            int hiY  = lowY + filter.getHeight();
            
            for (int j = lowY; j <= hiY; ++j)
            {
                for (int i = lowX; i <= hiX; ++i)
                {
                    sum += input.get(i, j, inputZ) * 
                        //filter.get(i - lowX, j - lowY, filterZ);
                        filter.get(j - lowY, i - lowX, filterZ);
                }
            }
            
            // Add the sum to the current value in this cell
            output.add(x, y, outputZ, sum);
        }
    }
}

void ConvLayer::feed(const vector<double>& x, vector<double>& y)
{
    Tensor3D input((vector<double>&) x, 0, mInputWidth, mInputHeight, mInputChannels);
    
    // Make sure the output is the correct size
    size_t outputWidth  = getOutputWidth();
    size_t outputHeight = getOutputHeight();
    size_t outputDepth  = mNumFilters;
    y.resize(outputWidth * outputHeight * outputDepth);
    
    // Wrap the important vectors in Tensor3D objects so we can work with them
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
    
void ConvLayer::calculateDeltas(Layer* downstream)
{
    vector<double>& downstreamDeltas = downstream->getDeltas();
    std::fill(downstreamDeltas.begin(), downstreamDeltas.end(), 0.0);
    
    size_t outputRows       = getOutputHeight();
    size_t outputCols       = getOutputWidth();
    size_t numFilterWeights = mFilterSize * mFilterSize * mInputChannels + 1;
    
    Tensor3D currentDeltas(mDeltas, 0, outputCols, outputRows, mNumFilters);
    Tensor3D targetDeltas(downstreamDeltas, 0, mInputWidth, mInputHeight, mInputChannels);
    
    for (size_t k = 0; k < mNumFilters; ++k)
    {
        Tensor3D currentFilter(*mParameters, mParametersStartIndex + (k * numFilterWeights), 
            mFilterSize, mFilterSize, mInputChannels);
        
        //size_t outChannelOffset = k * outputRows * outputCols;
        for (size_t j = 0; j < outputRows; ++j)
        {
            //size_t outRowOffset = j * outputCols;
            int inRowOffset     = j * mStride - mZeroPadding;
            for (size_t i = 0; i < outputCols; ++i)
            {
                //size_t index    = outChannelOffset + outRowOffset + i;
                int inColOffset = i * mStride - mZeroPadding;
                for (size_t z = 0; z < mInputChannels; ++z)
                {
                    //size_t kernelChannelOffset = z * mFilterSize * mFilterSize;
                    //size_t inChannelOffset     = z * mInputHeight * mInputWidth;
                    for (size_t y = 0; y < mFilterSize; ++y)
                    {
                        //size_t kernelRowOffset = y * mFilterSize;
                        int inRow              = inRowOffset + y;
                        for (size_t x = 0; x < mFilterSize; ++x)
                        {
                            int inCol = inColOffset + x;
                            if (inRow >= 0 && inRow < (int)mInputHeight && inCol >= 0 && inCol < (int) mInputWidth)
                            {
                                //size_t idx = inChannelOffset + mInputWidth * inRow + inCol;
                                
                                double val = currentFilter.get(x, y, z) * currentDeltas.get(i, j, k);
                                targetDeltas.add(inCol, inRow, z, val);
                                //downstreamDeltas[idx] += mKernels[k][kernelChannelOffset + kernelRowOffset + x] * mDeltas[index];
                            }
                        }
                    }
                }
            }
        }
    }
    
    // TODO: Implement. Convolve blame with filters to get deltas for previous
    // layer. deltas_l = weights_l+1 * deltas_l+1 * actD(net_l)
//    vector<double>& downstreamDeltas = downstream->getDeltas();
//    std::fill(downstreamDeltas.begin(), downstreamDeltas.end(), 0.0);
//
//    size_t numFilterWeights = mFilterSize * mFilterSize * mInputChannels + 1;
//    Activation& act = downstream->getActivationFunction();
//    
//    Tensor3D currentDeltas(mDeltas, 0, getOutputWidth(), getOutputHeight(), mNumFilters);
//    Tensor3D targetDeltas(downstreamDeltas, 0, mInputWidth, mInputHeight, mInputChannels);
//    
//    size_t padding = (mFilterSize - 1) / 2;
//    for (size_t channel = 0; channel < mInputChannels; ++channel)
//    {
//        for (size_t filter = 0; filter < mNumFilters; ++filter)
//        {
//            Tensor3D currentFilter(*mParameters, mParametersStartIndex + (filter * numFilterWeights), 
//                mFilterSize, mFilterSize, mInputChannels);
//            convolve2(currentDeltas, filter, padding, 1, currentFilter, channel, targetDeltas, channel);
//        }
//    }
    
    
//    
//    for (size_t i = 0; i < mInputs; ++i)
//    {
//        double actDerivative = (*act.second)
//            (downstream->getNet()[i], downstream->getActivation()[i]);
//
//        for (size_t j = 0; j < mOutputs; ++j)
//        {
//            size_t index  = mParametersStartIndex + (j * mInputs + i);
//            double weight = (*mParameters)[index];
//            downstreamDeltas[i] += weight * mDeltas[j] * actDerivative;
//        }
//    }
}

void ConvLayer::calculateDeltas(size_t outputIndex)
{
    // Blame is set to 0 for all outputs except the one we're interested in
    std::fill(mDeltas.begin(), mDeltas.end(), 0.0);
    
    // Apply the derivative of the activation function to the output of this
    // layer
    mDeltas[outputIndex] = (*mActFunction.second)
        (mNet[outputIndex], mActivation[outputIndex]);
}

void ConvLayer::calculateGradient(const vector<double>& x, size_t outputIndex, double* gradient)
{    
    // Calculate the output dimensions
    size_t numFilterWeights = mFilterSize * mFilterSize * mInputChannels + 1;
    size_t outputWidth      = getOutputWidth();
    size_t outputHeight     = getOutputHeight();
    size_t outputDepth      = mNumFilters;

    // Convert the output index into a 3D coordinate for the current output point
    size_t iz  = outputIndex / (outputWidth * outputHeight);
    size_t i2 = outputIndex % (outputWidth * outputHeight);
    size_t iy  = i2 / outputWidth;
    size_t ix  = i2 % outputWidth;
            
    // Wrap the important vectors in Tensor3D objects so we can work with them
    Tensor3D input((vector<double>&) x, 0, mInputWidth, mInputHeight, mInputChannels);
    Tensor3D deltas(mDeltas, 0, outputWidth, outputHeight, outputDepth);
      
    // Note: These loops can be run in any order. Each iteration is completely
    // independent of every other iteration.
    for (size_t filter = 0; filter < mNumFilters; ++filter)
    {
        Tensor3D grad(gradient + (filter * numFilterWeights), 0, 
            mFilterSize, mFilterSize, mInputChannels);
        
        // Convolve the appropriate region in the input with the deltas
        // in order to calculate the gradient for the weights
        for (size_t channel = 0; channel < mInputChannels; ++channel)
            convolve(input, ix, iy, iz, deltas, filter, grad, channel);
        
        // Sum over the deltas for this filter to calculate the bias gradient
        double bias = 0.0;
        for (size_t i = 0; i < outputWidth; ++i)
        {
            for (size_t j = 0; j < outputHeight; ++j)
                bias += deltas.get(i, j, filter);
        }
        size_t biasIndex    = numFilterWeights * (filter + 1) - 1;
        gradient[biasIndex] = bias;
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
    size_t w = (mInputWidth  - mFilterSize + 2 * mZeroPadding) / mStride + 1;
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

vector<double>& ConvLayer::getDeltas()
{
    return mDeltas;
}

size_t ConvLayer::getOutputWidth()
{
    return (mInputWidth - mFilterSize + 2 * mZeroPadding) / mStride + 1;
}

size_t ConvLayer::getOutputHeight()
{
    return (mInputHeight - mFilterSize + 2 * mZeroPadding) / mStride + 1;
}

size_t ConvLayer::getInputWidth()
{
    return mInputWidth;
}

size_t ConvLayer::getInputHeight()
{
    return mInputHeight;
}

size_t ConvLayer::getInputChannels()
{
    return mInputChannels;
}

size_t ConvLayer::getFilterSize()
{
    return mFilterSize;
}

size_t ConvLayer::getNumFilters()
{
    return mNumFilters;
}

size_t ConvLayer::getStride()
{
    return mStride;
}

size_t ConvLayer::getZeroPadding()
{
    return mZeroPadding;
}
    
Activation& ConvLayer::getActivationFunction()         
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

void NeuralNetwork::calculateDeltas(const size_t outputIndex)
{
    // Calculate the deltas on the last layer first
    mLayers.back()->calculateDeltas(outputIndex);
    
    // Apply the delta process recursively for each layer, moving backwards
    // through the network.
    for (int i = mLayers.size() - 1; i >= 1; --i)
        mLayers[i]->calculateDeltas(mLayers[i - 1]);
}

void NeuralNetwork::calculateGradientFromDeltas(const vector<double>& feature, 
    size_t outputIndex, vector<double>& gradient)
{
    const vector<double>* input = &feature;
    size_t weightIndex          = 0;
    
    for (size_t i = 0; i < mLayers.size(); ++i)
    {
        mLayers[i]->calculateGradient(*input, outputIndex, &gradient[weightIndex]);
        
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
        calculateGradientFromDeltas(x, i, jacobian[i]);
    }
}

// TODO: Figure out what the graident formula should be for a convolutional layer.
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

        // 3. Relate blame terms to the gradient
        vector<double>& deltas = mLayers.front()->getDeltas();
        int weightIndex        = 0;

        for (size_t j = 0; j < deltas.size(); ++j)
        {
            for (size_t i = 0; i < N; ++i)
                jacobian[k][i] += mParameters[weightIndex++] * deltas[j];
        }
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
/* 
 * File:   mnistConvolutionExample.cpp
 * Author: Jon C. Hammer
 *
 * Created on September 11, 2016, 9:21 AM
 */

#include <iostream>
#include "athena/Layer.h"
#include "athena/NeuralNetwork.h"
#include "athena/GradientDescent.h"
#include "athena/RMSProp.h"
#include "athena/SSEFunction.h"
#include "athena/CategoricalErrorFunction.h"
#include "athena/Matrix.h"
#include "athena/DataNormalizer.h"
#include "athena/ActivationFunction.h"
#include "athena/PrettyPrinter.h"
#include "athena/BatchIterator.h"
#include "athena/Timer.h"

using namespace std;

double variance(const Matrix& matrix, size_t column)
{
    // Calculate the variance
    double sum  = 0.0;
    double mean = matrix.columnMean(column);
    for (size_t i = 0; i < matrix.rows(); ++i)
    {
        double temp = matrix[i][column] - mean;
        sum += temp * temp;
    }
    
    return sum / (matrix.rows() - 1);
}

int main()
{
    // Load the data
    cout << "Loading data..." << endl;
    Matrix trainFeatures, trainLabels, testFeatures, testLabels;
    trainFeatures.loadARFF("../../data/mnist_train_features.arff");
    trainLabels.loadARFF("../../data/mnist_train_labels.arff");
    testFeatures.loadARFF("../../data/mnist_test_features.arff");
    testLabels.loadARFF("../../data/mnist_test_labels.arff");
    
    cout << "Train samples: " << trainFeatures.rows() << endl;
    cout << "Test samples:  " << testFeatures.rows()  << endl;
    cout << "Data loaded!" << endl;
    
    // Process the data
    cout << "Preprocessing the data..." << endl;
    scaleAllColumns(trainFeatures, 0.0, 255.0, -1.0, 1.0);
    scaleAllColumns(testFeatures,  0.0, 255.0, -1.0, 1.0);
    //normalizeVarianceAllColumns(trainFeatures);
    //normalizeVarianceAllColumns(testFeatures);
    trainLabels = convertColumnToOneHot(trainLabels, 0);
    testLabels  = convertColumnToOneHot(testLabels, 0);
    cout << "Data ready!" << endl;
    
    
    
    // Create a testing network
    FeedforwardLayer layer1(784, 80);
    FeedforwardLayer layer2(80, 30);
    FeedforwardLayer layer3(30, 10);
    
    //FeedforwardLayer layer1(784, 300);
    //Convolutional2DLayer layer1(28, 28, 1, 3, 32, 3, 1);
//    layer1.setActivationFunction(reluActivation);
    //Convolutional2DLayer layer2(layer1.getOutputWidth(), layer1.getOutputHeight(), layer1.getNumFilters(), 3, 32, 3, 1);
//    layer2.setActivationFunction(reluActivation);
    //FeedforwardLayer layer3(layer2.getOutputs(), 10);
    //layer3.setActivationFunction(reluActivation);
    
    NeuralNetwork network;
    network.addLayer(&layer1);
    network.addLayer(&layer2);
    network.addLayer(&layer3);
    network.initializeParameters();
    
    printf("%zu x %zu\n", layer1.getInputs(), layer1.getOutputs());
    printf("%zu x %zu\n", layer2.getInputs(), layer2.getOutputs());
    printf("%zu x %zu\n", layer3.getInputs(), layer3.getOutputs());
    cout << "# Parameters: " << network.getParameters().size() << endl;
    
    //layer1.normalizeKernels();
    //layer2.normalizeKernels();
    
    // Create a trainer
    SSEFunction<NeuralNetwork> errorFunc(network);
    CategoricalErrorFunction<NeuralNetwork> misclassifications(network);
    
    GradientDescent<NeuralNetwork> trainer(&errorFunc);
    trainer.setLearningRate(0.01);
    
    
    /*RMSProp<NeuralNetwork> trainer(&errorFunc);
    trainer.setLearningRate(0.001);
    trainer.setDecay(0.9);
    trainer.setMomentum(0.0001);*/
    
    BatchIterator it(trainFeatures, trainLabels, 1);
    Matrix* batchFeatures;
    Matrix* batchLabels;
    
    printf("%5d: %0.2f\t%.0f\t%f\n", 
        0,
        0.0,
        misclassifications.evaluate(testFeatures, testLabels), 
        errorFunc.evaluate(testFeatures, testLabels));
    cout.flush();
    
    Timer timer;
    
    for (size_t i = 0; i < 10; ++i)
    {
        
        while (it.hasNext())
        {
            it.lock(batchFeatures, batchLabels);
            trainer.iterate(*batchFeatures, *batchLabels);
            it.unlock();
        }
        it.reset();
        //trainer.iterate(trainFeatures, trainLabels);
        
        //printVector(network.getParameters(), 4);
        //layer1.normalizeKernels();
        //layer2.normalizeKernels();
        //printVector(network.getParameters(), 4);
        
        printf("%5zu: %0.2f\t%.0f\t%f\n", 
            (i+1),
            timer.getElapsedTimeSeconds(),
            misclassifications.evaluate(testFeatures, testLabels), 
            errorFunc.evaluate(testFeatures, testLabels));
        cout.flush();
        
        //trainer.setLearningRate(trainer.getLearningRate() * 0.9);
    }

    return 0;
}

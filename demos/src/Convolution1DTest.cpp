#include <iostream>
#include <vector>
#include "opkit/opkit.h"

using std::cout;
using std::endl;
using std::vector;

using namespace opkit;
using Type = float;

int main()
{
    Rand rand(42);

    // Create the network
    const size_t BATCH_SIZE     = 20;
    const size_t INPUT_SIZE     = 28;
    const size_t INPUT_CHANNELS = 2;
    const size_t FILTER_SIZE    = 3;
    const size_t NUM_FILTERS    = 2;
    const size_t PADDING        = 1;
    const size_t STRIDE         = 2;

    NeuralNetwork<Type> nn(BATCH_SIZE);
    Convolutional1DLayer<Type>* layer1 = new Convolutional1DLayer<Type>
        (INPUT_SIZE, INPUT_CHANNELS, FILTER_SIZE, NUM_FILTERS, PADDING, STRIDE);
    Convolutional1DLayer<Type>* layer2 = new Convolutional1DLayer<Type>
        (layer1->getOutputSize(), layer1->getOutputChannels(), 3, 32, 1, 1);
    Convolutional1DLayer<Type>* layer3 = new Convolutional1DLayer<Type>
        (layer2->getOutputSize(), layer2->getOutputChannels(), 3, 16, 1, 2);
    ActivationLayer<Type>* act1 = new ActivationLayer<Type>(
        layer3->getOutputs(), new tanhActivation<Type>());
    FullyConnectedLayer<Type>* layer4 = new FullyConnectedLayer<Type>(layer3->getOutputs(), 10);
    SoftmaxLayer<Type>* act2 = new SoftmaxLayer<Type>(layer4->getOutputs());

    // Convolutional2DLayer<Type>* layer1 = new Convolutional2DLayer<Type>
    //     (INPUT_SIZE, INPUT_SIZE, INPUT_CHANNELS, FILTER_SIZE, FILTER_SIZE,
    //      NUM_FILTERS, PADDING, PADDING, STRIDE, STRIDE);
    // Convolutional2DLayer<Type>* layer2 = new Convolutional2DLayer<Type>
    //     (layer1->getOutputWidth(), layer1->getOutputHeight(),
    //      layer1->getOutputChannels(), 2, 2, 32, 0, 0, 1, 1);
    // Convolutional2DLayer<Type>* layer3 = new Convolutional2DLayer<Type>
    //     (layer2->getOutputWidth(), layer2->getOutputHeight(),
    //      layer2->getOutputChannels(), 3, 3, 16, 1, 1, 2, 2);
    // ActivationLayer<Type>* act1 = new ActivationLayer<Type>(
    //     layer3->getOutputs(), new tanhActivation<Type>());
    // FullyConnectedLayer<Type>* layer4 = new FullyConnectedLayer<Type>(layer3->getOutputs(), 10);
    // SoftmaxLayer<Type>* act2 = new SoftmaxLayer<Type>(layer4->getOutputs());

    nn.addLayer(layer1);
    nn.addLayer(layer2);
    nn.addLayer(layer3);
    nn.addLayer(act1);
    nn.addLayer(layer4);
    nn.addLayer(act2);

    nn.print(cout, "");
    nn.initializeParameters(rand);
    // cout << "Filters: " << endl;
    // printVector(cout, nn.getParameters());

    for (size_t it = 0; it < 1; ++it)
    {
        // Create some test data
        vector<Type> x(nn.getInputs());
        for (size_t i = 0; i < x.size(); ++i)
            x[i] = rand.nextReal(0.0, 10.0);
        // cout << "x: ";
        // printVector(cout, x);

        // Evaluate the jacobian using both the exact approach and the finite
        // differences approximation.
        Matrix<Type> jacobianParameters, jacobianParameters2;
        nn.calculateJacobianParameters(x.data(), jacobianParameters);
        nn.Function::calculateJacobianParameters(x.data(), jacobianParameters2);

        for (size_t i = 0; i < jacobianParameters2.getRows(); ++i)
        {
            for (size_t j = 0; j < jacobianParameters2.getCols(); ++j)
            {
                if (std::abs(jacobianParameters(i, j) - jacobianParameters2(i, j)) > 1E-3)
                {
                    cout << "Parameters - Test Failed." << endl;
                    cout << "Exact: " << endl;
                    printMatrix(cout, jacobianParameters);

                    cout << "Approximate: " << endl;
                    printMatrix(cout, jacobianParameters2);
                    return 1;
                }
            }
        }
        cout << "Parameters - Test Passed." << endl;

        // Evaluate the jacobian using both the exact approach and the finite
        // differences approximation.
        Matrix<Type> jacobianInputs, jacobianInputs2;
        nn.calculateJacobianInputs(x.data(), jacobianInputs);
        nn.Function::calculateJacobianInputs(x.data(), jacobianInputs2);

        for (size_t i = 0; i < jacobianInputs2.getRows(); ++i)
        {
            for (size_t j = 0; j < jacobianInputs2.getCols(); ++j)
            {
                if (std::abs(jacobianInputs(i, j) - jacobianInputs2(i, j)) > 1E-3)
                {
                    cout << "Inputs - Test Failed." << endl;
                    cout << "Exact: " << endl;
                    printMatrix(cout, jacobianInputs);

                    cout << "Approximate: " << endl;
                    printMatrix(cout, jacobianInputs2);
                    return 1;
                }
            }
        }
        cout << "Inputs - Test Passed." << endl;

        // cout << "Exact: " << endl;
        // printMatrix(cout, jacobianInputs);
        //
        // cout << "Approximate: " << endl;
        // printMatrix(cout, jacobianInputs2);
    }

    return 0;
}

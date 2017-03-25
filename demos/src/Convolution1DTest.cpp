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
    // size_t WIDTH         = 4;
    // size_t HEIGHT        = 3;
    // size_t CHANNELS      = 2;
    // size_t KERNEL_WIDTH  = 2;
    // size_t KERNEL_HEIGHT = 2;
    // size_t X_PADDING     = 0;
    // size_t Y_PADDING     = 0;
    // size_t X_STRIDE      = 1;
    // size_t Y_STRIDE      = 1;
    // size_t X_DILATION    = 1;
    // size_t Y_DILATION    = 1;
    //
    // size_t OUT_WIDTH  = (WIDTH  + 2 * X_PADDING - (X_DILATION * (KERNEL_WIDTH  - 1) + 1)) / X_STRIDE + 1;
    // size_t OUT_HEIGHT = (HEIGHT + 2 * Y_PADDING - (Y_DILATION * (KERNEL_HEIGHT - 1) + 1)) / Y_STRIDE + 1;
    //
    // size_t ITERATIONS = 1;
    //
    // Matrix<Type> data(HEIGHT * CHANNELS, WIDTH);
    // for (size_t i = 0; i < data.getRows() * data.getCols(); ++i)
    // {
    //     data.data()[i] = (i + 1);
    // }
    // cout << "orig: " << endl;
    // printMatrix(cout, data);
    // cout << endl;
    //
    // // Caffe im2col
    // Matrix<Type> cols(KERNEL_WIDTH * KERNEL_HEIGHT * CHANNELS, OUT_WIDTH * OUT_HEIGHT);
    // im2col(data.data(), WIDTH, HEIGHT, CHANNELS,
    //     KERNEL_WIDTH, KERNEL_HEIGHT, X_PADDING, Y_PADDING,
    //     X_STRIDE, Y_STRIDE, X_DILATION, Y_DILATION, cols.data());
    // cout << "im2col(orig):" << endl;
    // printMatrix(cout, cols);
    // cout << endl;
    //
    // // My im2Row
    // Matrix<Type> rows(OUT_WIDTH * OUT_HEIGHT, KERNEL_WIDTH * KERNEL_HEIGHT * CHANNELS);
    // im2Row(data.data(), WIDTH, HEIGHT, CHANNELS,
    //     KERNEL_WIDTH, KERNEL_HEIGHT,
    //     X_PADDING, Y_PADDING,
    //     X_STRIDE, Y_STRIDE, rows.data());
    // cout << "im2row(orig):" << endl;
    // printMatrix(cout, rows);
    // cout << endl;
    //
    // // Caffe col2im
    // Matrix<Type> caffeReconstruction(HEIGHT * CHANNELS, WIDTH);
    // col2im_cpu(cols.data(), CHANNELS, HEIGHT, WIDTH,
    //     KERNEL_HEIGHT, KERNEL_WIDTH, Y_PADDING, X_PADDING,
    //     Y_STRIDE, X_STRIDE, Y_DILATION, X_DILATION, caffeReconstruction.data());
    // cout << "col2im(im2col(orig)):" << endl;
    // printMatrix(cout, caffeReconstruction);
    // cout << endl;
    //
    // // My row2Im
    // Matrix<Type> reconstruction(HEIGHT * CHANNELS, WIDTH);
    // row2Im(rows.data(), WIDTH, HEIGHT, CHANNELS,
    //     KERNEL_WIDTH, KERNEL_HEIGHT,
    //     X_PADDING, Y_PADDING, X_STRIDE, Y_STRIDE, reconstruction.data());
    // cout << "row2im(im2Row(orig)):" << endl;
    // printMatrix(cout, reconstruction);
    // cout << endl;
    Rand rand(42);

    // Create the network
    const size_t BATCH_SIZE     = 1;
    const size_t INPUT_SIZE     = 6;
    const size_t INPUT_CHANNELS = 2;
    const size_t FILTER_SIZE    = 3;
    const size_t NUM_FILTERS    = 2;
    const size_t PADDING        = 0;
    const size_t STRIDE         = 1;

    NeuralNetwork<Type> nn(BATCH_SIZE);
    // Convolutional1DLayer<Type>* layer1 = new Convolutional1DLayer<Type>
    //     (INPUT_SIZE, INPUT_CHANNELS, FILTER_SIZE, NUM_FILTERS, PADDING, STRIDE);
    // Convolutional1DLayer<Type>* layer2 = new Convolutional1DLayer<Type>
    //     (layer1->getOutputSize(), layer1->getOutputChannels(), 3, 32, 1, 1);
    // Convolutional1DLayer<Type>* layer3 = new Convolutional1DLayer<Type>
    //     (layer2->getOutputSize(), layer2->getOutputChannels(), 3, 16, 1, 1);
    // FullyConnectedLayer<Type>* layer4 = new FullyConnectedLayer<Type>(layer3->getOutputs(), 10);

    Convolutional2DLayer<Type>* layer1 = new Convolutional2DLayer<Type>
        (INPUT_SIZE, INPUT_SIZE, INPUT_CHANNELS, FILTER_SIZE, FILTER_SIZE,
         NUM_FILTERS, PADDING, PADDING, STRIDE, STRIDE);
    Convolutional2DLayer<Type>* layer2 = new Convolutional2DLayer<Type>
        (layer1->getOutputWidth(), layer1->getOutputHeight(),
         layer1->getOutputChannels(), 2, 2, 32, 0, 0, 1, 1);
    Convolutional2DLayer<Type>* layer3 = new Convolutional2DLayer<Type>
        (layer2->getOutputWidth(), layer2->getOutputHeight(),
         layer2->getOutputChannels(), 2, 2, 16, 0, 0, 1, 1);
    FullyConnectedLayer<Type>* layer4 = new FullyConnectedLayer<Type>(layer3->getOutputs(), 10);

    nn.addLayer(layer1);
    nn.addLayer(layer2);
    nn.addLayer(layer3);
    nn.addLayer(layer4);
    nn.print(cout, "");
    nn.initializeParameters(rand);
    // cout << "Filters: " << endl;
    // printVector(cout, nn.getParameters());

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
    return 0;
}

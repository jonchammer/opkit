#ifndef CONVOLUTIONAL_REGULARIZER_H
#define CONVOLUTIONAL_REGULARIZER_H

// #include "CostFunction.h"
// #include "Matrix.h"
// #include "NeuralNetwork.h"

// Returns the sign of the value [-1, 0, or 1] without using any branches
// template <typename T> int sign(T val)
// {
//     return (T(0) < val) - (val < T(0));
// }
//
// namespace opkit
// {
//
// template <class T>
// class ConvolutionalRegularizer : public CostFunction<T, NeuralNetwork<T>>
// {
// public:
//     using CostFunction<T, Model>::mBaseFunction;
//     constexpr static T DEFAULT_LAMBDA = 0.001;
//
//     ConvolutionalRegularizer(NeuralNetwork<T>& baseFunction,
//         size_t filterSize, const T lambda = DEFAULT_LAMBDA) :
//         CostFunction<T, Model>(baseFunction),
//         mFilterSize(filterSize), mLambda(lambda)
//     {
//         // Do nothing
//     }
//
//     T evaluate(const Matrix<T>& /*features*/, const Matrix<T>& /*labels*/)
//     {
//         T sum{};
//
//         for (size_t i = 0; i < mRelevantLayerIndices.size(); ++i)
//         {
//             const size_t layer               = mRelevantLayerIndices[i];
//             FullyConnectedLayer<T>* curLayer = (FullyConnectedLayer<T>*) mBaseFunction.getLayer(layer);
//             const T* params                  = curLayer->getParameters();
//             const size_t numParams           = curLayer->getNumParameters();
//
//             const size_t inputs  = curLayer->getInputs();
//             const size_t outputs = curLayer->getOutputs();
//
//             vector<T> means(mFilterSize);
//
//             for (int y = 0; y < outputs; ++y)
//             {
//                 for (int x = 0; x < inputs; ++x)
//                 {
//                     // Inside the convolutional window
//                     if (std::abs(y - x) < mFilterSize/2)
//                     {
//                         // Add to the means vector
//                         means[y - x + mFilterSize/2] += params[y * inputs + x];
//                     }
//
//                     // Outside the convolutional window. Apply L1 regularization
//                     else
//                     {
//                         sum += std::abs(params[y * inputs + x]);
//                     }
//                 }
//             }
//
//             // 2nd pass - Sum variances inside the convolutional windows
//             for (int y = 0; y < outputs; ++y)
//             {
//                 for (int x = 0; x < inputs; ++x)
//                 {
//                     if (std::abs(y - x) < mFilterSize/2)
//                     {
//                         T delta = params[y * inputs + x] - means[y - x + mFilterSize/2]);
//                         sum += delta * delta;
//                     }
//                 }
//             }
//         }
//
//         return mLambda * sum;
//         // T* params      = mBaseFunction.getParameters().data();
//         // const size_t N = mBaseFunction.getNumParameters();
//         //
//         // T sum{};
//         // for (size_t i = 0; i < N; ++i)
//         //     sum += std::abs(params[i]);
//         //
//         // return mLambda * sum;
//
//         // T* params      = mBaseFunction.getParameters().data();
//         // const size_t N = mBaseFunction.getNumParameters();
//         //
//         // T sum{};
//         // for (size_t i = 0; i < N; ++i)
//         //     sum += params[i] * params[i];
//         //
//         // return mLambda * sum;
//     }
//
//     void calculateGradientInputs(const Matrix<T>& features,
//         const Matrix<T>& labels, vector<T>& gradient)
//     {
//         // Regularization doesn't affect the gradient with respect to
//         // the inputs
//         std::fill(gradient.begin(), gradient.end(), T{});
//     }
//
//     void calculateGradientParameters(const Matrix<T>& features,
//         const Matrix<T>& labels, vector<T>& gradient)
//     {
//         T* grad              = gradient.data();
//         size_t curLayerIndex = 0;
//
//         for (size_t i = 0; i < mBaseFunction.getNumLayers(); ++i)
//         {
//             if (i == mRelevantLayerIndices[curLayerIndex])
//             {
//                 FullyConnectedLayer<T>* curLayer = (FullyConnectedLayer<T>*) mBaseFunction.getLayer(layer);
//                 const T* params                  = curLayer->getParameters();
//                 const size_t numParams           = curLayer->getNumParameters();
//
//                 const size_t inputs  = curLayer->getInputs();
//                 const size_t outputs = curLayer->getOutputs();
//
//                 vector<T> means(mFilterSize);
//
//                 for (int y = 0; y < outputs; ++y)
//                 {
//                     for (int x = 0; x < inputs; ++x)
//                     {
//                         // Inside the convolutional window
//                         if (std::abs(y - x) < mFilterSize/2)
//                         {
//                             // Add to the means vector
//                             means[y - x + mFilterSize/2] += params[y * inputs + x];
//                         }
//
//                         // Outside the convolutional window. Apply L1 regularization
//                         else
//                         {
//                             grad[y * inputs + x] = sign(params[y * inputs + x]) * mLambda;
//                         }
//                     }
//                 }
//
//                 // 2nd pass - Sum variances inside the convolutional windows
//                 for (int y = 0; y < outputs; ++y)
//                 {
//                     for (int x = 0; x < inputs; ++x)
//                     {
//                         if (std::abs(y - x) < mFilterSize/2)
//                         {
//                             grad[y * inputs + x] = (params[y * inputs + x] -
//                                 means[y - x + mFilterSize/2]) * mLambda;
//                         }
//                     }
//                 }
//
//                 curLayerIndex++;
//             }
//
//             grad += mBaseFunction.getLayer(i)->getNumParameters();
//         }
//
//         // T* grad        = gradient.data();
//         // T* params      = mBaseFunction.getParameters().data();
//         // const size_t N = mBaseFunction.getNumParameters();
//         //
//         // L1
//         // for (size_t i = 0; i < N; ++i)
//         //     grad[i] = sign(params[i]) * mLambda;
//         //
//         // L2
//         // for (size_t i = 0; i < N; ++i)
//         //     grad[i] = params[i] * mLambda;
//     }
//
// private:
//     size_t mFilterSize;
//     T mLambda;
//     vector<size_t> mRelevantLayerIndices;
// };
//
// }

#endif

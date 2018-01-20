#include <iostream>
#include "opkit/opkit.h"
#include "Plot.h"

using namespace std;
using namespace opkit;

// Generate some synthetic training data
template <class T>
void createData(Tensor<T>& x, Tensor<T>& y, const size_t N)
{
    Rand rand(42);
    
    x.resize({N, 1});
    y.resize({N, 1});
    for (size_t i = 0; i < N; ++i)
    {
        T val        = rand.nextReal(-10.0f, 10.0f);
        x.at({i, 0}) = val;
        y.at({i, 0}) = 2.0f * val * val - 0.3f * val - 0.7f + rand.nextGaussian(0.0f, 0.01f);
    }
}

// Samples y(x) for several different values of x. Returns a tensor with all (x, y) pairs.
template <class T>
Tensor<T> sample(Graph<T>& y, T min, T max, T interval)
{
    const size_t N = ceil((max - min) / interval) + 1;
    Tensor<T> res({N, 2});
    
    Graph<T>& x = *y.find("x");
    T val = min;
    for (size_t i = 0; i < N; ++i)
    {
        x.assign(Tensor<T>::fromValues({val}, {1, 1}));
        res.at({i, 0}) = val;
        res.at({i, 1}) = T(y());
        val += interval;
    }
    
    return res;
}

int main()
{
    using T = float;
    
    // Generate the training data
    Tensor<T> trainFeatures, trainLabels;
    createData(trainFeatures, trainLabels, 50);
    
    // Create the graph variables
    Rand rand(42);
    auto x  = make_variable<T>("x",  zeroes<T>({1, 1}));
    auto y1 = make_variable<T>("y",  zeroes<T>({1, 1}));
    auto w1 = make_variable<T>("w1", xavier<T>({1, 10}, rand));
    auto b1 = make_variable<T>("b1", zeroes<T>({1, 10}));
    auto w2 = make_variable<T>("w2", xavier<T>({10, 1}, rand));
    auto b2 = make_variable<T>("b2", zeroes<T>({1, 1}));
    
    // Create the model itself and the error function
    auto y   = linear(tanh(linear(x, w1, b1)), w2, b2);
    auto err = sse(y, y1);
    
    // Set up a trainer to update the graph variables
    T lr = 1E-5;
    auto trainer = gradientDescent(err, {"w1", "b1", "w2", "b2"}, lr);
    auto decay   = multBy(*trainer.find("lr"), make_constant<T>(0.9999));
    
    // Function to plot curves as images to see how training progresses
    auto plot = [](Graph<T>& y, size_t i)
    {
        Tensor<T> points = sample(y, -10.0f, 10.0f, 0.5f);
        string filename = "out/" + to_string(i) + ".png";
        if (!plotScatter(filename, points, 0, 1, 512, 256, -10.0f, 10.0f, -10.0f, 200.0f))
            cout << "Unable to save: " << filename << endl;
    };
    plot(y, 0);
    
    // Repeatedly refine the model
    for (int i = 0; i < 100000; ++i)
    {
        x.assign(trainFeatures);
        y1.assign(trainLabels);
        trainer();
        decay();
        
        if (i % 1000 == 0)
            plot(y, i);
    }
 
    return 0;
}
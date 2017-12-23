#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <iostream>
#include <fstream>
#include "tensor/Tensor.h"
#include "util/Assert.h"

namespace opkit
{

// Saves the given tensor to a binary file under the given name. The file can be
// read using loadTensorRaw().
template <class T>
bool saveTensorRaw(const std::string& filename, const Tensor<T>& tensor)
{
    // Open the file
    std::ofstream dout(filename.c_str(), std::ios::binary);
    if (!dout) return false;

    // Write the metadata
    size_t rank = tensor.rank();
    dout.write((char*) &rank, sizeof(size_t));
    for (size_t i = 0; i < rank; ++i)
    {
        size_t dim = tensor.shape(i);
        dout.write((char*) &dim, sizeof(size_t));
    }

    // Write the actual data
    for (auto& it : tensor)
        dout.write((char*) &it, sizeof(T));

    dout.close();
    return true;
}

// Loads a binary file created by saveTensorRaw() into the given tensor. Note
// that this tensor will be resized, so any connection to other tensors or
// Storage objects will be severed.
template <class T>
bool loadTensorRaw(const std::string& filename, Tensor<T>& tensor)
{
    // Open the file
    std::ifstream din(filename.c_str(), std::ios::binary);
    if (!din) return false;

    // Read the metadata and set up the tensor
    size_t numElements = 1;
    size_t rank        = 0;
    din.read((char*) &rank, sizeof(size_t));

    vector<size_t> shape(rank);
    for (size_t i = 0; i < rank; ++i)
    {
        din.read((char*) &shape[i], sizeof(size_t));
        numElements *= shape[i];
    }
    tensor.resize(shape.begin(), shape.end());

    // Read the actual data
    T* data = tensor.data();
    din.read((char*) data, sizeof(T) * numElements);
    din.close();
    return true;
}

template <class T>
bool saveTensorARFF(const std::string& filename, const Tensor<T>& tensor)
{
    ASSERT(tensor.rank() == 2, "Only rank-2 tensors are currently supported.");

    // Open the file
    std::ofstream dout(filename.c_str());
    if (!dout) return false;

    // Write the metadata
    dout.precision(10);
    dout << "@RELATION Tensor\n";
    for (size_t i = 0; i < tensor.shape(1); ++i)
        dout << "@ATTRIBUTE a_" << i << " REAL\n";

    // Write the data
    dout << "@DATA\n";
    auto it = tensor.begin();
    while (it != tensor.end())
    {
        for (size_t i = 0; i < tensor.shape(1) - 1; ++i)
        {
            dout << *it << ',';
            ++it;
        }
        dout << *it << '\n';
        ++it;
    }

    dout.close();
    return true;
}

template <class T>
bool loadTensorARFF(const std::string& filename, Tensor<T>& tensor)
{
    // Open the file
    std::ifstream din(filename.c_str());
    if (!din) return false;

    std::vector<double> data;
    std::string line;
    size_t columns = 0;
    while (getline(din, line))
    {
        // Trim the string from the left side
        line.erase(line.begin(), std::find_if(line.begin(), line.end(), [](int ch)
        {
            return !std::isspace(ch);
        }));
        if (line.empty()) continue;

        // Downcase the string
        for (auto& c : line)
            c = std::tolower(c);

        // Count the number of columns
        if (line.find("@attribute") != std::string::npos)
            ++columns;

        // Ignore all other data until we find the @data line.
        if (line.find("@data") != std::string::npos)
        {
            while (din.peek() != EOF)
            {
                for (size_t c = 0; c < columns - 1; ++c)
                {
                    getline(din, line, ',');
                    data.push_back(stof(line));
                }
                getline(din, line);
                data.push_back(stof(line));
            }
        }
    }

    // Copy the data into the tensor
    tensor.resize( {data.size() / columns, columns} );
    std::copy(data.begin(), data.end(), tensor.data());
    din.close();
    return true;
}

}

#endif

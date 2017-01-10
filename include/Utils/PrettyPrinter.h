/*
 * File:   PrettyPrinter.h
 * Author: Jon
 *
 * Created on August 4, 2016, 10:06 AM
 */

#ifndef PRETTY_PRINTER_H
#define PRETTY_PRINTER_H

#include <iostream>
#include <iomanip>
#include <vector>
#include "Matrix.h"
#include "Dataset.h"
#include "Tensor3D.h"

namespace opkit
{

template <class T>
void printVector(const std::vector<T>& vec, size_t decimalPlaces = 2)
{
    std::cout << std::fixed << std::showpoint << std::setprecision(decimalPlaces);
    std::cout << "[";
    for (size_t i = 0; i < vec.size() - 1; ++i)
        std::cout << vec[i] << ", ";

    if (vec.size() >= 1)
        std::cout << vec.back();

    std::cout << "]" << std::endl;
}

template <class T>
void printDataset(const Dataset<T>& mat, size_t decimalPlaces = 2, size_t width = 6)
{
    std::cout << std::fixed << std::showpoint << std::setprecision(decimalPlaces);

    for (size_t j = 0; j < mat.rows(); ++j)
    {
        std::cout << "[";
        for (size_t i = 0; i < mat.cols() - 1; ++i)
            std::cout << std::setw(width) << mat[j][i] << ", ";

        if (mat[j].size() >= 1)
            std::cout << std::setw(width) << mat[j].back();

        std::cout << "]" << std::endl;
    }
}

template <class T>
void printMatrix(const Matrix<T>& mat, size_t decimalPlaces = 2, size_t width = 6)
{
    std::cout << std::fixed << std::showpoint << std::setprecision(decimalPlaces);
    const size_t N = mat.getRows();
    const size_t M = mat.getCols();

    for (size_t j = 0; j < N; ++j)
    {
        std::cout << "[";
        for (size_t i = 0; i < M - 1; ++i)
            std::cout << std::setw(width) << mat(j, i) << ", ";

        if (M >= 1)
            std::cout << std::setw(width) << mat(j, M - 1) << "]" << std::endl;
    }
}

// Prints a given 1D vector as if it were a 3D tensor. The front face is printed
// first, followed by the next face, and so on.
template <class T>
void print3DTensor(const std::vector<T>& vec, size_t width, size_t height,
    size_t depth, size_t decimalPlaces = 2, size_t spacing = 6)
{
    std::cout << std::fixed << std::showpoint << std::setprecision(decimalPlaces);

    size_t i = 0;
    for (size_t l = 0; l < depth; ++l)
    {
        for (size_t j = 0; j < height; ++j)
        {
            std::cout << "[";
            for (size_t k = 0; k < width - 1; ++k)
                std::cout << std::setw(spacing) << vec[i++] << ", ";

            if (width >= 1)
                std::cout << std::setw(spacing) << vec[i++];

            std::cout << "]" << std::endl;
        }
        std::cout << std::endl;
    }
}

template <class T>
void print3DTensor(const Tensor3D<T>& tensor, size_t decimalPlaces = 2,
    size_t spacing = 6)
{
    const size_t width  = tensor.getWidth();
    const size_t height = tensor.getHeight();
    const size_t depth  = tensor.getDepth();

    std::cout << std::fixed << std::showpoint << std::setprecision(decimalPlaces);

    for (size_t l = 0; l < depth; ++l)
    {
        for (size_t j = 0; j < height; ++j)
        {
            std::cout << "[";
            for (size_t k = 0; k < width - 1; ++k)
                std::cout << std::setw(spacing) << tensor.get(k, j, l) << ", ";

            if (width >= 1)
                std::cout << std::setw(spacing) << tensor.get(width - 1, j, l);

            std::cout << "]" << std::endl;
        }
        std::cout << std::endl;
    }
}

};

#endif /* PRETTY_PRINTER_H */

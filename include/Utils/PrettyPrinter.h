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
#include "Tensor3D.h"

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

void printMatrix(const Matrix& mat, size_t decimalPlaces = 2, size_t width = 6);

// Prints a given 1D vector as if it were a 3D tensor. The front face is printed
// first, followed by the next face, and so on.
void print3DTensor(const std::vector<double>& vec, size_t width, size_t height, 
    size_t depth, size_t decimalPlaces = 2, size_t spacing = 6);

void print3DTensor(const Tensor3D& tensor, size_t decimalPlaces = 2, size_t width = 6);

#endif /* PRETTY_PRINTER_H */


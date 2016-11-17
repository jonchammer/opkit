#include "PrettyPrinter.h"

namespace athena
{

void printMatrix(const Matrix& mat, size_t decimalPlaces, size_t width)
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

void print3DTensor(const std::vector<double>& vec, size_t width, size_t height, 
    size_t depth, size_t decimalPlaces, size_t spacing)
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

void print3DTensor(const Tensor3D& tensor, size_t decimalPlaces, size_t spacing)
{    
    size_t width  = tensor.getWidth();
    size_t height = tensor.getHeight();
    size_t depth  = tensor.getDepth();

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
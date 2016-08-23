#include "PrettyPrinter.h"

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
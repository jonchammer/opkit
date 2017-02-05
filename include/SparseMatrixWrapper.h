#ifndef SPARSE_MATRIX_WRAPPER
#define SPARSE_MATRIX_WRAPPER

#include <vector>
#include <unordered_map>
#include <random>
#include <iostream>
#include "Matrix.h"

using std::cout;
using std::endl;

namespace opkit
{

// Maps x -> Index
typedef std::unordered_map<size_t, size_t> SparseVector;

template <class T>
class SparseMatrixWrapper
{
public:
    SparseMatrixWrapper(const size_t rows, const size_t cols) :
        mData(nullptr), mRows(rows), mCols(cols), mIndices(rows)
    {}

    void clear()
    {
        for (size_t r = 0; r < mRows; ++r)
            mIndices[r].clear();
    }

    void set(const size_t row, const size_t col, size_t index)
    {
        mIndices[row].insert( {col, index} );
    }

    bool isSet(const size_t row, const size_t col) const
    {
        return mIndices[row].find(col) != mIndices[row].end();
    }

    T get(const size_t row, const size_t col) const
    {
        auto it = mIndices[row].find(col);
        return it != mIndices[row].end() ? mData[it->second] : T{};
    }

    // Multiply this sparse matrix by a dense vector (x), and put the result
    // in the dense vector y.
    void multiply(const T* x, T* y)
    {
        // Calculate y = mData * x
        std::fill(y, y + mRows, T{});
        for (size_t r = 0; r < mRows; ++r)
        {
            for (auto it = mIndices[r].begin(); it != mIndices[r].end(); ++it)
            {
                size_t c = it->first;
                size_t i = it->second;
                y[r]    += mData[i] * x[c];
            }
        }
    }

    // Multiply the transpose of this sparse matrix by a dense vector (x),
    // and put the result in the dense vector y.
    void multiplyTranspose(const T* x, T* y)
    {
        // Calculate y = mData^T * x
        std::fill(y, y + mCols, T{});
        for (size_t r = 0; r < mRows; ++r)
        {
            for (auto it = mIndices[r].begin(); it != mIndices[r].end(); ++it)
            {
                size_t c = it->first;
                size_t i = it->second;
                y[c]    += mData[i] * x[r];
            }
        }
    }

    // Multiply the dense vectors x and y together (to generate the outer
    // product between the two.) Technically, the result should be dense,
    // but we will throw away those connections that aren't present in this
    // sparse matrix.
    // NOTE: This operation will add the outer product to whatever is already
    // present in A.
    void outerProduct(const T* x, const T* y, T* A)
    {
        for (size_t r = 0; r < mRows; ++r)
        {
            for (auto it = mIndices[r].begin(); it != mIndices[r].end(); ++it)
            {
                size_t c = it->first;
                size_t i = it->second;
                A[i]    += x[r] * y[c];
            }
        }
    }

    Matrix<T> toMatrix() const
    {
        Matrix<T> res(mRows, mCols);

        for (size_t r = 0; r < mRows; ++r)
        {
            for (auto it = mIndices[r].begin(); it != mIndices[r].end(); ++it)
            {
                size_t c  = it->first;
                size_t i  = it->second;
                res(r, c) = mData[i];
            }
        }

        return res;
    }

    void setData(T* data)
    {
        mData = data;
    }

    size_t getRows() const
    {
        return mRows;
    }

    size_t getCols() const
    {
        return mCols;
    }

private:
    T* mData;
    size_t mRows, mCols;
    std::vector<SparseVector> mIndices;

    // void test()
    // {
    //     // Initialize m1
    //     Matrix<double> m1(2, 5);
    //     m1(0, 0) = 1.0; m1(0, 3) = 4.0;
    //     m1(1, 1) = 7.0; m1(1, 2) = 8.0; m1(1, 4) = 10.0;
    //
    //     // Initialize m2
    //     SparseMatrixWrapper<double> m2(2, 5);
    //     m2.set(0, 0, 0); m2.set(0, 3, 1);
    //     m2.set(1, 1, 2); m2.set(1, 2, 3); m2.set(1, 4, 4);
    //     double data[] = {1.0, 4.0, 7.0, 8.0, 10.0};
    //     m2.setData(data);
    //
    //     printMatrix(m1);       cout << endl;
    //     printSparseMatrix(m2); cout << endl;
    //
    //     // Check multiplication
    //     Matrix<double> x(5, 1, {1.0, 2.0, 3.0, 4.0, 5.0});
    //     Matrix<double> y(m1 * x);
    //     printMatrix(y); cout << endl;
    //
    //     y.fill();
    //     m2.multiply(x.data(), y.data());
    //     printMatrix(y); cout << endl;
    //
    //     // Check transposed multiplication
    //     Matrix<double> x2(2, 1, {1.0, 2.0});
    //     Matrix<double> y2(transpose(m1) * x2);
    //     printMatrix(y2); cout << endl;
    //
    //     y2.fill();
    //     m2.multiplyTranspose(x2.data(), y2.data());
    //     printMatrix(y2); cout << endl;
    //
    //     // Check outer product
    //     Matrix<double> y3(x2 * transpose(x));
    //     printMatrix(y3); cout << endl;
    //
    //     y3.fill();
    //     double outerProductResult[5];
    //     m2.outerProduct(x2.data(), x.data(), outerProductResult);
    //     m2.setData(outerProductResult);
    //     printSparseMatrix(m2); cout << endl;
    // }
};

}

#endif

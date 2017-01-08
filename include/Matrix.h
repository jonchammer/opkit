#ifndef MATRIX_H
#define MATRIX_H

#include <cstdlib>
#include <vector>
using std::vector;

namespace opkit
{

template <class T>
class Matrix
{
public:

    // Default Constructor
    Matrix() : mRows(0), mCols(0) {}

    // Non-default Constructor
    Matrix(const size_t rows, const size_t cols) :
        mData(rows * cols), mRows(rows), mCols(cols) {}

    // Copy constructor
    Matrix(Matrix& other) :
        mData(other.mData), mRows(other.mRows), mCols(other.mCols) {}

    // Returns a contiguous array that is used internally to store
    // the contents of the matrix.
    T* data()
    {
        return mData.data();
    }

    // Returns a contiguous array that is used internally to store
    // the contents of the matrix.
    const T* data() const
    {
        return mData.data();
    }

    // Allows the user to access a given cell of the matrix
    T& operator() (const size_t row, const size_t col)
    {
        return mData[row * mCols + col];
    }

    // Allows the user to access a given cell of the matrix
    const T& operator() (const size_t row, const size_t col) const
    {
        return mData[row * mCols + col];
    }

    // Returns a pointer to the beginning of the given row.
    T* operator() (const size_t row)
    {
        return mData.data() + (row * mCols);
    }

    // Returns a pointer to the beginning of the given row.
    const T* operator() (const size_t row) const
    {
        return mData.data() + (row * mCols);
    }

    // Updates the size of the matrix. No guarantees are made about
    // the contents of the matrix after this operation.
    void resize(const size_t rows, const size_t cols)
    {
        mData.resize(rows * cols);
        mRows = rows;
        mCols = cols;
    }

    // Sets every cell in the matrix to the given value. The default
    // value is 0 (for whatever type is being used).
    void fill(const T value = T{})
    {
        std::fill(mData.begin(), mData.end(), value);
    }

    // Copy a given subsection of 'other' into this matrix. 'rowBegin',
    // 'colBegin', 'rowCount', and 'colCount' specify where the source data
    // is relative to 'source'. 'destRowBegin' and 'destColBegin' specify where
    // the data should go in this matrix.
    void copy(const Matrix<T> source,
        const size_t rowBegin, const size_t colBegin,
        const size_t rowCount, const size_t colCount,
        const size_t destRowBegin = 0, const size_t destColBegin = 0)
    {
        for (size_t row = 0; row < rowCount; ++row)
        {
            for (size_t col = 0; col < colCount; ++col)
            {
                size_t destRow = destRowBegin + row;
                size_t destCol = destColBegin + col;
                size_t srcRow  = rowBegin + row;
                size_t srcCol  = colBegin + col;

                mData[destRow * mCols + destCol] = source(srcRow, srcCol);
            }
        }
    }

    // Returns the number of rows in the matrix
    size_t getRows() const
    {
        return mRows;
    }

    // Returns the number of columns in the matrix
    size_t getCols() const
    {
        return mCols;
    }

private:
    vector<T> mData;     // Stores the actual matrix data
    size_t mRows, mCols; // Stores the matrix dimensions
};

};

#endif

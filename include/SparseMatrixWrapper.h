#ifndef SPARSE_MATRIX_WRAPPER
#define SPARSE_MATRIX_WRAPPER

#include <vector>
#include <unordered_set>
#include <random>
#include <iostream>
#include "Rand.h"
#include "Matrix.h"

using std::cout;
using std::endl;

namespace opkit
{

// This class interprets the elements of an array of numbers as the non-zero
// entries in a sparse matrix. It should be used primarily for arithmetic
// operations (e.g. multiplying a sparse matrix by a dense vector), as opposed
// to random access operations.
//
// Internally, this class uses the coordinate-wise
// storage principle to allow for fast access of elements either by row or by
// column. Two extra arrays are stored, rowIndices and colIndices. rowIndices[i]
// is the row index for cell i in the data array. colIndices[i] is the column
// index for cell i in the data array.
template <class T>
class SparseMatrixWrapper
{
public:

    // Construct a new Sparse Matrix Wrapper. None of the connections are set.
    //
    // @param rows           - The number of rows in the sparse matrix.
    // @param cols           - The number of columns in the sparse matrix.
    // @param numConnections - The number of non-zero elements in the matrix.
    SparseMatrixWrapper(const size_t rows, const size_t cols,
        const size_t numConnections) :
        mData(nullptr), mRows(rows), mCols(cols),
        mNumConnections(numConnections), mRowIndices(new size_t[numConnections]),
        mColIndices(new size_t[numConnections])
    {}

    // Construct a new Sparse Matrix Wrapper where all of the connections are
    // set. Note that this does tend to defeat the purpose of having a sparse
    // matrix, but it's useful in some circumstances.
    //
    // @param rows           - The number of rows in the sparse matrix.
    // @param cols           - The number of columns in the sparse matrix.
    // @param numConnections - The number of non-zero elements in the matrix.
    SparseMatrixWrapper(const size_t rows, const size_t cols) :
        mData(nullptr), mRows(rows), mCols(cols),
        mNumConnections(rows * cols), mRowIndices(new size_t[rows * cols]),
        mColIndices(new size_t[rows * cols])
    {
        size_t i = 0;
        for (size_t r = 0; r < rows; ++r)
        {
            for (size_t c = 0; c < cols; ++c)
            {
                mRowIndices[i] = r;
                mColIndices[i] = c;
                i++;
            }
        }
    }

    // Construct a new Sparse Matrix Wrapper. A total of 'numConnections'
    // connections will be set randomly based on the given Random number
    // generator.
    //
    // @param rows           - The number of rows in the sparse matrix.
    // @param cols           - The number of columns in the sparse matrix.
    // @param numConnections - The number of non-zero elements in the matrix.
    // @param rand           - Source of randomness for the connections.
    SparseMatrixWrapper(const size_t rows, const size_t cols,
        const size_t numConnections, Rand<size_t>& rand) :
        mData(nullptr), mRows(rows), mCols(cols),
        mNumConnections(numConnections), mRowIndices(new size_t[numConnections]),
        mColIndices(new size_t[numConnections])
    {
        // Used to keep track of which cells we have already filled
        auto cellHash = [](const std::pair<size_t, size_t>& p)
        {
            return std::hash<size_t>()(p.first) ^
                   std::hash<size_t>()(p.second);
        };
        std::unordered_set<std::pair<size_t, size_t>, decltype(cellHash)>
        mIndices(mNumConnections, cellHash);

        size_t i = 0;
        for (size_t j = 0; j < mNumConnections; ++j)
        {
            // Pick a random row and column
            size_t r, c;
            do
            {
                r = rand(0, rows - 1);
                c = rand(0, cols - 1);
            }
            while(mIndices.find({r, c}) != mIndices.end());

            // Set this cell
            mIndices.insert( {r, c} );
            mRowIndices[i] = r;
            mColIndices[i] = c;
            i++;
        }
    }

    // Destructor
    ~SparseMatrixWrapper()
    {
        delete[] mRowIndices;
        delete[] mColIndices;
        mRowIndices = nullptr;
        mColIndices = nullptr;
    }

    // Attach the given cell of the matrix to the corresponding index in the
    // underlying data array.
    void set(const size_t row, const size_t col, const size_t index)
    {
        mRowIndices[index] = row;
        mColIndices[index] = col;
    }

    // Returns true if the given cell is part of the sparse matrix.
    //
    // NOTE: The way this particular data structure is designed, checking for
    // the existance of a cell is not trivial. Therefore, we have to iterate
    // over all of the connections to find a match. This is not something that
    // should be done in a tight loop, since it carries an O(n) runtime.
    bool isSet(const size_t row, const size_t col) const
    {
        for (size_t i = 0; i < mNumConnections; ++i)
        {
            if (mRowIndices[i] == row && mColIndices[i] == col)
                return true;
        }
        return false;
    }

    // Multiply this sparse matrix by a dense vector (x), and put the result
    // in the dense vector y.
    void multiply(const T* x, T* y) const
    {
        // Calculate y = mData * x
        std::fill(y, y + mRows, T{});
        for (size_t i = 0; i < mNumConnections; ++i)
        {
            y[mRowIndices[i]] += mData[i] * x[mColIndices[i]];
        }
    }

    // Multiply the transpose of this sparse matrix by a dense vector (x),
    // and put the result in the dense vector y.
    void multiplyTranspose(const T* x, T* y) const
    {
        // Calculate y = mData^T * x
        std::fill(y, y + mCols, T{});
        for (size_t i = 0; i < mNumConnections; ++i)
        {
            y[mColIndices[i]] += mData[i] * x[mRowIndices[i]];
        }
    }

    // Multiply the dense vectors x and y together (to generate the outer
    // product between the two.) Technically, the result should be dense,
    // but we will throw away those connections that aren't present in this
    // sparse matrix.
    //
    // NOTE: This operation will add the outer product to whatever is already
    // present in A.
    void outerProduct(const T* x, const T* y, T* A) const
    {
        for (size_t i = 0; i < mNumConnections; ++i)
        {
            A[i] += x[mRowIndices[i]] * y[mColIndices[i]];
        }
    }

    // Creates and returns a dense Matrix that has the same content as this
    // sparse matrix. This call involves allocating new storage for the dense
    // matrix and copying the appropriate values from the sparse matrix.
    Matrix<T> toMatrix() const
    {
        Matrix<T> res(mRows, mCols);
        for (size_t i = 0; i < mNumConnections; ++i)
            res(mRowIndices[i], mColIndices[i]) = mData[i];

        return res;
    }

    // Assign an underlying data array that holds the non-zero elements of this
    // matrix. This must be called before any of the arithmetic operations can
    // be performed.
    void setData(T* data)
    {
        mData = data;
    }

    // Simple getters
    size_t getRows()           const { return mRows;           }
    size_t getCols()           const { return mCols;           }
    size_t getNumConnections() const { return mNumConnections; }

private:
    T* mData;
    size_t mRows, mCols, mNumConnections;
    size_t* mRowIndices;
    size_t* mColIndices;

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

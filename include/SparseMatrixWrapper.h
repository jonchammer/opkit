#ifndef SPARSE_MATRIX_WRAPPER
#define SPARSE_MATRIX_WRAPPER

#include <vector>
#include <unordered_map>
#include <random>
#include <iostream>
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
        mData(nullptr), mRows(rows), mCols(cols), mIndices(rows) {}

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

    // Multiply this sparse matrix by a dense vector (x), and add the result
    // to the dense vector y.
    void multiply(const T* x, T* y)
    {
        // Calculate y += mData * x
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
    // and add the result to the dense vector y.
    void multiplyTranspose(const T* x, T* y)
    {
        // Calculate y += mData^T * x
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
};

}

#endif

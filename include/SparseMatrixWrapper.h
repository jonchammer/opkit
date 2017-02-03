#ifndef SPARSE_MATRIX_WRAPPER
#define SPARSE_MATRIX_WRAPPER

#include <vector>
#include <unordered_map>
#include <random>

namespace opkit
{

// Maps x -> Index
typedef std::unordered_map<size_t, size_t> SparseVector;

template <class T>
class SparseMatrixWrapper
{
public:
    SparseMatrixWrapper(T* data, const size_t rows,
        const size_t cols, const size_t numConnections) :
        mData(data), mRows(rows), mCols(cols),
        mIndices(rows)
    {
        // Assign rows and columns for each connection
        if (rows * cols == numConnections)
        {
            size_t i = 0;
            for (size_t r = 0; r < rows; ++r)
            {
                for (size_t c = 0; c < cols; ++c)
                {
                    mIndices[r].insert( {c, i} );
                    ++i;
                }
            }
        }

        else
        {
            std::default_random_engine generator;
            std::uniform_int_distribution<int> rRows(0, rows);
            std::uniform_int_distribution<int> rCols(0, cols);

            size_t i = 0;
            for (size_t j = 0; j < numConnections; ++j)
            {
                // Pick a random row and column
                size_t r, c;
                do
                {
                    r = rRows(generator);
                    c = rCols(generator);
                }
                while(mIndices[r].find(c) != mIndices[r].end());

                // Tie this index to the given row and column
                mIndices[r].insert( {c, i} );
                ++i;
            }
        }
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

private:
    T* mData;
    size_t mRows, mCols;
    std::vector<SparseVector> mIndices;
};

}

#endif

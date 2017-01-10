#ifndef MATRIX_H
#define MATRIX_H

#include <cstdlib>
#include <vector>
#include <initializer_list>
#include "Acceleration.h"
using std::vector;

namespace opkit
{

template <class T>
class Matrix
{
public:

    // Default Constructor
    Matrix() : mRows(0), mCols(0) {}

    // Non-default Constructors
    Matrix(const size_t rows, const size_t cols) :
        mData(rows * cols), mRows(rows), mCols(cols) {}

    Matrix(const size_t rows, const size_t cols, std::initializer_list<T> list) :
        mData(list), mRows(rows), mCols(cols) {}

    // Vector constructors - The first version copies the contents of 'data'
    // into this matrix. The second moves the contents, which is much cheaper.
    Matrix(vector<T>& data)  : mData(data), mRows(1), mCols(data.size()) {}
    Matrix(vector<T>&& data) : mData(data), mRows(1), mCols(data.size()) {}

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

    // Swap the contents of this matrix with 'other'
    void swap(const Matrix<T>& other)
    {
        mData.swap(other.mData);
        swap(mRows, other.mRows);
        swap(mCols, other.mCols);
    }

    // Swaps the data in this matrix with the contents of the given buffer.
    // This operation DOES NOT change the size of the matrix. It merely provides
    // an efficient mechanism for temporarily 'wrapping' a vector. If a
    // permanent wrapping is desired, consider using the vector move constructor
    // instead.
    void swap(vector<double>& other)
    {
        mData.swap(other);
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

// Multiplication machinery
template <class LHS, class RHS>
struct Multiplication
{
    Multiplication(const LHS& _lhs, const RHS& _rhs) :
        lhs(_lhs), rhs(_rhs) {}

    const LHS& lhs;
    const RHS& rhs;
};

template <class LHS, class RHS>
Multiplication<LHS, RHS> operator*(const LHS& lhs, const RHS& rhs)
{
    return Multiplication<LHS, RHS>(lhs, rhs);
}

template <class T>
void apply(Multiplication<Matrix<T>, Matrix<T>> base, Matrix<T>& target)
{
    const Matrix<T>& m1 = base.lhs;
    const Matrix<T>& m2 = base.rhs;
    target.resize(m1.getRows(), m2.getCols());
    mmMultiply(m1.data(), m2.data(), target.data(),
        target.getRows(), target.getCols(), m1.getCols(), 1.0, 1.0);
}

// Addition machinery
template <class LHS, class RHS>
struct Addition
{
    Addition(const LHS& _lhs, const RHS& _rhs) :
        lhs(_lhs), rhs(_rhs) {}

    const LHS& lhs;
    const RHS& rhs;
};

template <class LHS, class RHS>
Addition<LHS, RHS> operator+(const LHS& lhs, const RHS& rhs)
{
    return Addition<LHS, RHS>(lhs, rhs);
}

template <class T>
void apply(Addition<Matrix<T>, Matrix<T>> base, Matrix<T>& target)
{
    const Matrix<T>& m1 = base.lhs;
    const Matrix<T>& m2 = base.rhs;
    const size_t M = m1.getRows();
    const size_t N = m1.getCols();

    target.resize(M, N);
    vAdd(m1.data(), target.data(), M * N);
    vAdd(m2.data(), target.data(), M * N);
}

// Allowing multiplication by scalars
// scalar * matrix
template <class T>
void apply(Multiplication<T, Matrix<T>> base, Matrix<T>& target)
{
    const Matrix<T>& m = base.rhs;
    target.resize(m.getRows(), m.getCols());
    vAdd(m.data(), target.data(), m.getRows() * m.getCols(), base.lhs);
}

// matrix * scalar
template <class T>
void apply(Multiplication<Matrix<T>, T> base, Matrix<T>& target)
{
    const Matrix<T>& m = base.lhs;
    target.resize(m.getRows(), m.getCols());
    vAdd(m.data(), target.data(), m.getRows() * m.getCols(), base.rhs);
}

// Multiplication<?, ?> * scalar
template
<
    class T,
    class LHS,
    class RHS,
    template <class LHS, class RHS> class ChildOp
>
void apply(Multiplication<ChildOp<LHS, RHS>, T> base, Matrix<T>& target)
{
    Matrix<T> temp;
    apply(base.lhs, temp);
    apply(Multiplication<Matrix<T>, T>(temp, base.rhs), target);
}

// scalar * Multiplication<?, ?>
template
<
    class T,
    class LHS,
    class RHS,
    template <class LHS, class RHS> class ChildOp
>
void apply(Multiplication<T, ChildOp<LHS, RHS>> base, Matrix<T>& target)
{
    Matrix<T> temp;
    apply(base.rhs, temp);
    apply(Multiplication<T, Matrix<T>>(base.lhs, temp), target);
}

// General recursive machinery. Allows us to combine matrices and expressions to
// evaluate any arbitrary expression. Attempts are made to reduce the number
// of dynamic allocations, but some are unavoidable.
// Recursive Case 1 - the left branch is a complex expression.
template
<
    class T,
    class LHS,
    class RHS,
    template <class LHS, class RHS> class ChildOp,
    template <class ChildOp, class M> class ParentOp
>
void apply(ParentOp<ChildOp<LHS, RHS>, Matrix<T>> base, Matrix<T>& target)
{
    // Complicated case - create a temporary matrix, evaluate the lhs of the
    // tree (to populate the temporary matrix), then do a simple evaluation
    // between two matrices (temp and base.rhs).
    // NOTE: temp is created on the stack, but it will live long enough for
    // the second call to apply() to finish, so it doesn't create a problem.
    Matrix<T> temp;
    apply(base.lhs, temp);
    apply(ParentOp<Matrix<T>, Matrix<T>>(temp, base.rhs), target);
}

// Recursive Case 2 - the right branch is a complex expression.
template
<
    class T,
    class LHS,
    class RHS,
    template <class LHS, class RHS> class ChildOp,
    template <class ChildOp, class M> class ParentOp
>
void apply(ParentOp<Matrix<T>, ChildOp<LHS, RHS>> base, Matrix<T>& target)
{
    // Complicated case - create a temporary matrix, evaluate the rhs of the
    // tree (to populate the temporary matrix), then do a simple evaluation
    // between two matrices (base.lhs and temp).
    // NOTE: temp is created on the stack, but it will live long enough for
    // the second call to apply() to finish, so it doesn't create a problem.
    Matrix<T> temp;
    apply(base.rhs, temp);
    apply(ParentOp<Matrix<T>, Matrix<T>>(base.lhs, temp), target);
}

// Recursive Case 3 - both branches are complex expressions.
template
<
    class T,
    class LHS1, class LHS2,
    class RHS1, class RHS2,
    template <class LHS1, class RHS1> class ChildOp1,
    template <class LHS2, class RHS2> class ChildOp2,
    template <class ChildOp1, class ChildOp2> class ParentOp
>
void apply(ParentOp<ChildOp1<LHS1, RHS1>, ChildOp2<LHS2, RHS2>> base, Matrix<T>& target)
{
    // Really complicated case - create two temporary matrices, evaluate each
    // side of the tree to populate them, then do a simple evaluation
    // between the two remaining matrices (temp1 and temp2).
    // NOTE: temp1 and temp2 are created on the stack, but they will live long
    // enough for the final call to apply() to finish, so it doesn't create a problem.
    Matrix<T> temp1;
    Matrix<T> temp2;
    apply(base.lhs, temp1);
    apply(base.rhs, temp2);
    apply(ParentOp<Matrix<T>, Matrix<T>>(temp1, temp2), target);
}

// Forced evaluation. When these functions are called, the abstract syntax tree
// is collapsed to create the final result.
template <class T, class Exp>
Matrix<T>& operator+=(Matrix<T>& lhs, Exp rhs)
{
    apply(rhs, lhs);
    return lhs;
}

};

#endif

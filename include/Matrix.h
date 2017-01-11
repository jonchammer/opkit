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
    void swap(Matrix<T>& other)
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
    void swap(vector<T>& other)
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

//----------------------------------------------------------------------------//

// Interior nodes of the AST that have two children
template <class LHS, class RHS, class Op>
struct BinaryExpression
{
    BinaryExpression(const LHS& lhs, const RHS& rhs) :
        left(lhs), right(rhs) {}

    template <class T>
    void apply(Matrix<T>& target) const
    {
        Op::apply(left, right, target);
    }

    const LHS& left;
    const RHS& right;
};

// Interior nodes of the AST that have one child
template <class Base, class Op>
struct UnaryExpression
{
    UnaryExpression(const Base& c) : child(c) {}

    template <class T>
    void apply(Matrix<T>& target) const
    {
        Op::apply(child, target);
    }

    const Base& child;
};

// Base class for all binary operations (e.g. addition, multiplication).
// These functions would have to be declared in every binary operation
// without this class, so it's mostly here to consolodate code a bit.
template <class Base>
struct BinaryOp
{
    // Expression on left side
    template <class T, class LHS>
    static void apply(const LHS& lhs, const Matrix<T>& rhs, Matrix<T>& target)
    {
        Matrix<T> temp;
        lhs.apply(temp);
        Base::apply(temp, rhs, target);
    }

    // Expression on right side
    template <class T, class RHS>
    static void apply(const Matrix<T>& lhs, const RHS& rhs, Matrix<T>& target)
    {
        Matrix<T> temp;
        rhs.apply(temp);
        Base::apply(lhs, temp, target);
    }

    // Expression on both sides
    template <class T, class LHS, class RHS>
    static void apply(const LHS& lhs, const RHS& rhs, Matrix<T>& target)
    {
        Matrix<T> temp1;
        Matrix<T> temp2;
        lhs.apply(temp1);
        rhs.apply(temp2);
        Base::apply(temp1, temp2, target);
    }
};

// Base class for all unary operations (e.g. transpose)
template <class Base>
struct UnaryOp
{
    template <class T, class E>
    static void apply(const E& e, Matrix<T>& target)
    {
        Matrix<T> temp;
        e.apply(temp);
        Base::apply(temp, target);
    }
};

// Operator implementations
struct Addition : public BinaryOp<Addition>
{
    // Required so that the compiler can find the other versions of
    // apply that are declared in BinaryOp.
    using BinaryOp<Addition>::apply;

    template <class T>
    static void apply(const Matrix<T>& m1,
        const Matrix<T>& m2, Matrix<T>& target)
    {
        const size_t M = m1.getRows();
        const size_t N = m1.getCols();

        target.resize(M, N);
        vAdd(m1.data(), target.data(), M * N);
        vAdd(m2.data(), target.data(), M * N);
    }
};

struct Subtraction : public BinaryOp<Subtraction>
{
    // Required so that the compiler can find the other versions of
    // apply that are declared in BinaryOp.
    using BinaryOp<Subtraction>::apply;

    template <class T>
    static void apply(const Matrix<T>& m1,
        const Matrix<T>& m2, Matrix<T>& target)
    {
        const size_t M = m1.getRows();
        const size_t N = m1.getCols();

        // y += A - B       -->
        // y = y + A - B    -->
        // y = y + A + (-B) -->
        // y = (y + A) + (-B)
        target.resize(M, N);
        vAdd(m1.data(), target.data(), M * N);
        vAdd(m2.data(), target.data(), M * N, -1.0);
    }
};

struct Multiplication : BinaryOp<Multiplication>
{
    // Required so that the compiler can find the other versions of
    // apply that are declared in BinaryOp.
    using BinaryOp<Multiplication>::apply;

    template <class T>
    static void apply(const Matrix<T>& m1, const Matrix<T>& m2, Matrix<T>& target)
    {
        target.resize(m1.getRows(), m2.getCols());
        mmMultiply(m1.data(), m2.data(), target.data(),
            target.getRows(), target.getCols(), m1.getCols(), 1.0, 1.0);
    }
};

struct ScalarMultiplication
{
    template <class T>
    static void apply(const T val, const Matrix<T>& m, Matrix<T>& target)
    {
        target.resize(m.getRows(), m.getCols());
        vAdd(m.data(), target.data(), m.getRows() * m.getCols(), val);
    }

    template <class T>
    static void apply(const Matrix<T>& m, const T val, Matrix<T>& target)
    {
        target.resize(m.getRows(), m.getCols());
        vAdd(m.data(), target.data(), m.getRows() * m.getCols(), val);
    }

    // Right side is an expression
    template <class T, class RHS>
    static void apply(const T val, const RHS& rhs, Matrix<T>& target)
    {
        Matrix<T> temp;
        rhs.apply(temp);
        apply(val, temp, target);
    }

    // Left side is an expression
    template <class T, class LHS>
    static void apply(const LHS& lhs, const T val, Matrix<T>& target)
    {
        Matrix<T> temp;
        lhs.apply(temp);
        apply(temp, val, target);
    }
};

struct Transpose : UnaryOp<Transpose>
{
    // Required so that the compiler can find the other versions of
    // apply that are declared in UnaryOp.
    using UnaryOp<Transpose>::apply;

    template <class T>
    static void apply(const Matrix<T>& m, Matrix<T>& target)
    {
        const size_t N = m.getRows();
        const size_t M = m.getCols();

        target.resize(M, N);
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                target(i, j) += m(j, i);
    }
};

// Operators - These are used to construct the AST at compile time.
template <class LHS, class RHS>
BinaryExpression<LHS, RHS, Addition>
operator+ (const LHS& lhs, const RHS& rhs)
{
    return BinaryExpression<LHS, RHS, Addition>(lhs, rhs);
}

template <class LHS, class RHS>
BinaryExpression<LHS, RHS, Subtraction>
operator- (const LHS& lhs, const RHS& rhs)
{
    return BinaryExpression<LHS, RHS, Subtraction>(lhs, rhs);
}

// Multiplication when both sides are not numbers (e.g. Matrix, Matrix)
// enable_if is used to control the return value using the SFINAE pattern.
// If the compile-time condition is true, the return type will be what we
// define. If not, there is no return type.
template <class LHS, class RHS>
typename std::enable_if
<
    !std::is_arithmetic<LHS>::value && !std::is_arithmetic<RHS>::value,
    BinaryExpression<LHS, RHS, Multiplication>
>::type
operator* (const LHS& lhs, const RHS& rhs)
{
    return BinaryExpression<LHS, RHS, Multiplication>(lhs, rhs);
}

// Multiplication when the LHS is a number, but the RHS is not.
template <class LHS, class RHS>
typename std::enable_if
<
    std::is_arithmetic<LHS>::value && !std::is_arithmetic<RHS>::value,
    BinaryExpression<LHS, RHS, ScalarMultiplication>
>::type
operator* (const LHS lhs, const RHS& rhs)
{
    return BinaryExpression<LHS, RHS, ScalarMultiplication>(lhs, rhs);
}

// Multiplication when the RHS is a number, but the LHS is not.
template <class LHS, class RHS>
typename std::enable_if
<
    !std::is_arithmetic<LHS>::value && std::is_arithmetic<RHS>::value,
    BinaryExpression<LHS, RHS, ScalarMultiplication>
>::type
operator* (const LHS& lhs, const RHS rhs)
{
    return BinaryExpression<LHS, RHS, ScalarMultiplication>(lhs, rhs);
}

// We could also overload an operator for this operation, but the function
// syntax seems clearer to me.
template <class Base>
UnaryExpression<Base, Transpose>
transpose (const Base& base)
{
    return UnaryExpression<Base, Transpose>(base);
}

// Forced evaluation. When these functions are called, the abstract syntax tree
// is collapsed to create the final result.
template <class T, class Exp>
Matrix<T>& operator+=(Matrix<T>& lhs, Exp rhs)
{
    rhs.apply(lhs);
    return lhs;
}

};

#endif

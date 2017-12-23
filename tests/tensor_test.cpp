#include <iostream>
#include <vector>
#include "opkit/opkit.h"

using namespace std;
using namespace opkit;

template <class Container>
void printContainer(const Container& container)
{
    cout << "[ ";
    for (const auto& elem : container)
        cout << elem << " ";
    cout << "]" << endl;
}

template <class T>
bool equals(const Tensor<T>& a, const Tensor<T>& b)
{
    if (a.size() != b.size()) return false;

    auto aIt = a.begin();
    auto bIt = b.begin();
    while (aIt != a.end())
    {
        if (*aIt != *bIt) return false;
        ++aIt;
        ++bIt;
    }
    return true;
}

// sub()
bool t1()
{
    Storage<double> storage(40);
    for (size_t i = 0; i < storage.size(); ++i)
        storage[i] = i;

    Tensor<double> test(storage, {2, 5, 4});
    Tensor<double> x = sub(test, {{1, 1}, {0, 3}, {2, 3}} );

    Tensor<double> truth(Storage<double>({22, 23, 26, 27, 30, 31, 34, 35}), {1, 4, 2} );
    return equals(x, truth);
}

bool t2()
{
// shape = <3, 4>, stride = <7, 1>, offset = 0

// 0  1  2  3 [ 4  5  6]
// 7  8  9 10 [11 12 13]
// 14 15 16 17 [18 19 20]

// ----------------------------------------------
// narrow(1, 1, 2)
// 1  2
// 8  9
// 15 16

// shape = <3, 2>, stride = <7, 1>, offset = 1
    return true;
}

// select()
bool t3()
{
    Storage<double> storage(40);
    for (size_t i = 0; i < storage.size(); ++i)
        storage[i] = i;

    Tensor<double> test(storage, {2, 5, 4});
    Tensor<double> x = select( test, 2, 1 );

    Tensor<double> truth(Storage<double>({1, 5, 9, 13, 17, 21, 25, 29, 33, 37}), {2, 5});
    return equals(x, truth);
}

// transpose
bool t4()
{
    Storage<double> storage(40);
    for (size_t i = 0; i < storage.size(); ++i)
        storage[i] = i;

    Tensor<double> test(storage, {2, 5, 4});
    Tensor<double> x = transpose(test, 0, 2);

    Storage<double> reference(
    {
        0, 20, 4, 24,  8, 28, 12, 32, 16, 36,
        1, 21, 5, 25,  9, 29, 13, 33, 17, 37,
        2, 22, 6, 26, 10, 30, 14, 34, 18, 38,
        3, 23, 7, 27, 11, 31, 15, 35, 19, 39
    });
    Tensor<double> truth(reference, {4, 5, 2});
    return equals(x, truth);
}

// permute()
bool t5()
{
    Storage<double> storage(40);
    for (size_t i = 0; i < storage.size(); ++i)
        storage[i] = i;

    Tensor<double> test(storage, {2, 5, 4});
    Tensor<double> x = permute(test, {2, 0, 1});

    Storage<double> reference(
    {
         0,  4,  8, 12, 16,
        20, 24, 28, 32, 36,

         1,  5,  9, 13, 17,
        21, 25, 29, 33, 37,

         2,  6, 10, 14, 18,
        22, 26, 30, 34, 38,

         3,  7, 11, 15, 19,
        23, 27, 31, 35, 39
    });
    Tensor<double> truth(reference, {4, 2, 5});
    return equals(x, truth);
}

bool t6()
{
    return true;
}

int main()
{
    int i = 1;
    for (auto& test : {t1, t2, t3, t4, t5, t6})
    {
        if (test())
            cout << "Test " << i << " passed." << endl;
        else cout << "Test " << i << " failed.   <---" << endl;

        ++i;
    }

    return 0;
}

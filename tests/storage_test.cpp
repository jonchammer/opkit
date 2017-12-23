#include <iostream>
#include <vector>
#include "opkit/opkit.h"

using namespace std;
using namespace opkit;

template <class T>
void print(const Storage<T>& storage, ostream& out = cout)
{
    for (auto& elem: storage)
        out << elem << " ";
    out << endl;
}

template <class T>
bool equals(const Storage<T>& storage, const std::vector<T>& vec)
{
    if (storage.size() != vec.size()) return false;
    for (size_t i = 0; i < storage.size(); ++i)
    {
        if (storage[i] != vec[i]) return false;
    }
    return true;
}

// Construction / fill / size
bool t1()
{
    Storage<double> x(10);
    x.fill(0.0);

    if (x.size() != 10) return false;

    return equals(x, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
}

// Resize / size / empty
bool t2()
{
    Storage<int> x;
    if (x.size() != 0 || !x.empty()) return false;
    x.resize(15);
    x.fill(0);

    if (x.size() != 15) return false;

    return equals(x, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
}

// Clone
bool t3()
{
    Storage<double> x(10);
    x.fill(2.7);

    Storage<int> y = x.clone<int>();
    if (y.size() != x.size()) return false;

    return equals(y, {2, 2, 2, 2, 2, 2, 2, 2, 2, 2});
}

// Simple Views
bool t4()
{
    Storage<double> x(10);
    Storage<double> y(x, 5);

    x.fill(5.0);
    y.fill(2.0);

    return equals(x, {5, 5, 5, 5, 5, 2, 2, 2, 2, 2}) &&
        equals(y, {2, 2, 2, 2, 2});
}

// Complex Views
bool t5()
{
    Storage<double> x(10);
    Storage<double> y(x, 5);
    Storage<double> z(y, 2, 2);

    x.fill(5.0);
    y.fill(2.0);
    z.fill(1.0);

    return equals(x, {5, 5, 5, 5, 5, 2, 2, 1, 1, 2}) &&
        equals(y, {2, 2, 1, 1, 2}) &&
        equals(z, {1, 1});
}

// Alternative constructors
bool t6()
{
    Storage<double> x({1, 2, 3, 4, 5});

    vector<int> values({6, 7, 8, 9, 10});
    Storage<int> y(values.begin(), values.end());

    return equals(x, {1, 2, 3, 4, 5}) && equals(y, {6, 7, 8, 9, 10});
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

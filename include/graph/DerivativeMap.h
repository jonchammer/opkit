#ifndef DERIVATIVE_MAP_H
#define DERIVATIVE_MAP_H

#include <unordered_map>
#include <unordered_set>
#include "graph/core/GraphAPI.h"

namespace opkit
{

// This class represents a mapping between function names (as used in the graph
// nodes) to specialized derivative functions. This class is a singleton, so use
// the static instance() function to obtain a valid class instance.
template <class T>
class DerivativeMap
{
private:
    using FunctionMap = std::unordered_map<std::string,
        std::function<void(Graph<T> node, Graph<T> delta,
            std::vector<Graph<T>>& gradients)>>;
    FunctionMap mMap;

    std::unordered_set<std::string> mNonDifferentiable;

    DerivativeMap() {}

public:
    // Remove constructors and operators used to create new instances
    DerivativeMap(const DerivativeMap& other)       = delete;
    DerivativeMap(DerivativeMap&& other)            = delete;
    DerivativeMap& operator=(DerivativeMap& other)  = delete;
    DerivativeMap& operator=(DerivativeMap&& other) = delete;

    // Instead, allow access to the single instance
    static DerivativeMap& instance()
    {
        static DerivativeMap mInstance;
        return mInstance;
    }

    // Add a new (name, function) pair to the map
    template <class Func>
    void add(const std::string& name, Func&& func)
    {
        mMap.emplace(name, std::forward<Func>(func));
    }

    // Add a new non-differentiable operation to the map
    void addNondifferentiable(const std::string& name)
    {
        mNonDifferentiable.emplace(name);
    }

    // Call the function currently associated to the given name if there is one.
    // When a non-differentiable function is provided, this function will return
    // false, but names that are not registered at all will cause an assertion
    // failure.
    template <class... Args>
    bool call(const std::string& name, Args&&... args)
    {
        if (mNonDifferentiable.find(name) != mNonDifferentiable.end())
            return false;
        else
        {
            // Ensure a derivative op has been registered
            ASSERT(mMap.find(name) != mMap.end(),
                "No derivative registered for \"" + name + "\"");

            mMap[name](std::forward<Args>(args)...);
            return true;
        }
    }
};

// Simpler function for user interface.
template <class T, class Func>
void registerDerivative(const std::string& name, Func&& derivativeFn)
{
    DerivativeMap<T>::instance().add(name, std::forward<Func>(derivativeFn));
}

template <class T>
void registerNonDifferentiable(const std::string& name)
{
    DerivativeMap<T>::instance().addNondifferentiable(name);
}

}

#endif

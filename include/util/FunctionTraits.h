// Adapted from: https://stackoverflow.com/questions/2562320/specializing-a-template-on-a-lambda-in-c0x
// This file provides a convenient system for pulling the return type and
// argument types from any function-like object, including normal functions,
// functors, and lambda functions.
//
// Example usage:
// using T = function_traits<decltype(lambda)>::result_type;

#ifndef FUNCTION_TRAITS_H
#define FUNCTION_TRAITS_H

#include <tuple>
#include <utility>
using std::tuple;
using std::declval;

// First, a convenient struct in which to store all the results:
template<bool is_method_, bool is_const_method_, typename C, typename R, typename ...Args>
struct function_traits_results {
    constexpr static bool is_method = is_method_;
    constexpr static bool is_const_method = is_const_method_;
    typedef C class_type; // void for plain functions. Otherwise,
                          // the functor/lambda type
    typedef R return_type;
    typedef tuple<Args...> args_type_as_tuple;
};

// This will extract all the details from a method-signature:
template<typename>
struct intermediate_step;
template<typename R, typename C, typename ...Args>
struct intermediate_step<R (C::*) (Args...)>  // non-const methods
    : public function_traits_results<true, false, C, R, Args...>
{
};
template<typename R, typename C, typename ...Args>
struct intermediate_step<R (C::*) (Args...) const> // const methods
    : public function_traits_results<true, true, C, R, Args...>
{
};

// These next two overloads do the initial task of separating
// plain function pointers for functors with ::operator()
template<typename R, typename ...Args>
function_traits_results<false, false, void, R, Args...>
function_traits_helper(R (*) (Args...) );
template<typename F, typename ..., typename MemberType = decltype(&F::operator()) >
intermediate_step<MemberType>
function_traits_helper(F);

// Finally, the actual `function_traits` struct, that delegates
// everything to the helper
template <typename T>
struct function_traits : public decltype(function_traits_helper( declval<T>() ) )
{
};

#endif

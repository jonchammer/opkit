#ifndef ASSERT_H
#define ASSERT_H

#include <iostream>
#include "StackTrace.h"

#ifndef NDEBUG
#   define ASSERT(condition, message)                                          \
    do {                                                                       \
        if (! (condition))                                                     \
        {                                                                      \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__   \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            print_stacktrace(stderr, 10);                                      \
            std::terminate();                                                  \
        }                                                                      \
    } while (false)
#else
#   define ASSERT(condition, message) do { } while (false)
#endif

#endif

#ifndef ASSERT_H
#define ASSERT_H

#include <iostream>
#include "StackTrace.h"

#ifndef NDEBUG
    #define ASSERT(condition, message)                                         \
    do {                                                                       \
        if (! (condition))                                                     \
        {                                                                      \
            std::cerr << "ASSERTION FAILURE"                   << '\n'         \
                      << "Condition: `" #condition "`"         << '\n'         \
                      << "Message:    " << message             << '\n'         \
                      << "Function:   " << __PRETTY_FUNCTION__ << '\n'         \
                      << "File:       " << __FILE__            << '\n'         \
                      << "Line:       " << __LINE__            << "\n\n";      \
            print_stacktrace(stderr, 10);                                      \
            std::terminate();                                                  \
        }                                                                      \
    } while (false)
#else
    #define ASSERT(condition, message) do { } while (false)
#endif

#endif

/* 
 * File:   CategoricalErrorFunction.h
 * Author: Jon C. Hammer
 *
 * Created on September 11, 2016, 10:17 AM
 */

#ifndef CATEGORICALERRORFUNCTION_H
#define CATEGORICALERRORFUNCTION_H

#include "ErrorFunction.h"
#include "Matrix.h"

class CategoricalErrorFunction : public ErrorFunction
{
public:
    CategoricalErrorFunction(Function& baseFunction);
    double evaluate(const Matrix& features, const Matrix& labels);
};
#endif /* CATEGORICALERRORFUNCTION_H */


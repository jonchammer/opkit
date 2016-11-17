/* 
 * File:   FunctionPersistence.h
 * Author: Jon C. Hammer
 *
 * Created on July 21, 2016, 4:31 PM
 */

#ifndef MODELPERSISTENCE_H
#define MODELPERSISTENCE_H

#include <iostream>
#include <fstream>
#include <vector>
#include "Function.h"

namespace athena
{

/**
 * Save a function to a plain text file with the given filename. The number of 
 * parameters will be written first, followed by each parameter on a new line.
 * 
 * @param filename. The name of the file to be written.
 * @param function. The function to be persisted.
 * @return true if the file is opened and written correctly, false otherwise.
 */
bool saveFunctionText(const std::string& filename, const Function& function)
{
    std::ofstream dout(filename.c_str());
    if (!dout) return false;
    
    const std::vector<double>& parameters = function.getParameters();
    
    // Write the number of parameters first
    dout << parameters.size() << endl;
    
    // Then write each of the parameter values
    for (size_t i = 0; i < parameters.size(); ++i)
        dout << parameters[i] << endl;
    
    dout.close();
    return true;
}

/**
 * Load a function from a plain text file with the given filename. The file must
 * be in the same format as used for saveFunctionText().
 * 
 * @param filename. The name of the file to be read.
 * @param function.    The destination of the parameters read from the file.
 * @return true if the file is opened and read correctly, false otherwise.
 */
bool loadFunctionText(const std::string& filename, Function& function)
{
    std::ifstream din(filename.c_str());
    if (!din) return false;
    
    std::vector<double>& parameters = function.getParameters();
    
    // Read the number of parameters first
    size_t numParams;
    din >> numParams;
    parameters.resize(numParams);
    
    // Then read each of the individual parameter values
    for (size_t i = 0; i < numParams; ++i)
        din >> parameters[i];
    
    din.close();
    return true;
}

/**
 * Save a function to a raw binary file with the given filename. The number of 
 * parameters will be written first, followed by each parameter.
 * 
 * @param filename. The name of the file to be written.
 * @param function.    The function to be persisted.
 * @return true if the file is opened and written correctly, false otherwise.
 */
bool saveFunctionBinary(const std::string& filename, const Function& function)
{
    std::ofstream dout(filename.c_str(), std::ios::binary);
    if (!dout) return false;
    
    const std::vector<double>& parameters = function.getParameters();
    
    // Write the number of parameters first
    size_t numParams = parameters.size();
    dout.write((char*) &numParams, sizeof(size_t));
    
    // Then write the parameter values
    dout.write((char*) &parameters[0], sizeof(double) * numParams);
    
    dout.close();
    return true;
}

/**
 * Load a function from a raw binary file with the given filename. The file must
 * be in the same format as used for saveFunctionBinary().
 * 
 * @param filename. The name of the file to be read.
 * @param function.    The destination of the parameters read from the file.
 * @return true if the file is opened and read correctly, false otherwise.
 */
bool loadFunctionBinary(const std::string& filename, Function& function)
{
    std::ifstream din(filename.c_str(), std::ios::binary);
    if (!din) return false;
    
    std::vector<double>& parameters = function.getParameters();
    
    // Read the number of parameters first
    size_t numParams = 0;
    din.read((char*) &numParams, sizeof(size_t));
    parameters.resize(numParams);
    
    // Then read the parameter values
    din.read((char*) &parameters[0], sizeof(double) * numParams);
    
    din.close();
    return true;
}

};

#endif /* MODELPERSISTENCE_H */


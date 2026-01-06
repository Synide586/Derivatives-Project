#pragma once
#include <string.h>
#include <iostream>
#include <regex>
#include <limits>
#include "../include/json.hpp"

namespace marketAPI
{
    int inputValidator(const std::string &s);
    std::string parse(int x);
    std::string authenticate();
    void help();
    std::string buy();
    std::string cancel();
    std::string modifyOrder();
    std::string getOrderbook();
    std::string getCurrentPositions();
    std::string subscribe();

}
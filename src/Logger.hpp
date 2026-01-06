#pragma once
#include <iostream>
#include <mutex>
#include <string>
#include "../include/json.hpp"


class Logger
{
public:
    static void logInfo(const std::string &message);
    static void logWarn(const std::string &message);
    static void logError(const std::string &message);
    static void logResult(const nlohmann::json &message);

private:
    static std::mutex mutex_;
};
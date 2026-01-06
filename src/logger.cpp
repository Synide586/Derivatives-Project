#include "Logger.hpp"


// The mutex takes ownership of the output stream, to ensure sequential delivery of output

std::mutex Logger::mutex_;

void Logger::logInfo(const std::string &message)
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::cout << "[INFO]: " << message << std::endl;
}


void Logger::logWarn(const std::string &message)
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::cerr << "[WARN]: " << message << std::endl;
}

void Logger::logError(const std::string &message)
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::cerr << "[ERROR]: " << message << std::endl;
}

void Logger::logResult(const nlohmann::json &message){
    std::lock_guard<std::mutex> lock(mutex_);
    nlohmann::json output;
    std::cout << "-------------------------------------------------------------------" << std::endl;
    if (message.contains("result")) {
        std::cout << "Success!" << std::endl << std:: endl;
        output["result"] = message["result"];
    }
    if (message.contains("error")){
        std::cout << "Error!" << std::endl << std:: endl;

        output["error"] = message["error"];
    }
    if (message.contains("usDiff")) {
        output["Response time(microseconds):"] = message["usDiff"];
    }
    std::cout << "Message Recieved" << output.dump(4) << std::endl;
    std::cout << "-------------------------------------------------------------------" << std::endl;

}

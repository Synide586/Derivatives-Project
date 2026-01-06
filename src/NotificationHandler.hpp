#pragma once
#include <queue>
#include <mutex>
#include "../include/json.hpp"
#include "Logger.hpp"

class NotificationHandler
{
public:
    explicit NotificationHandler(size_t maxCapacity);
    void addNotification(const nlohmann::json &notification);
    void displayNotifications();

private:
    std::queue<nlohmann::json> notifications_;
    std::mutex mutex_;
    size_t maxCapacity_;
};

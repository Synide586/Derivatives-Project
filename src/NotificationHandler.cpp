#include "NotificationHandler.hpp"

NotificationHandler::NotificationHandler(size_t maxCapacity) : maxCapacity_(maxCapacity) {}

/*
The mutex prevents a race condition when a new notification is recieved when a notification is being read
*/

void NotificationHandler::addNotification(const nlohmann::json &notification)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (notifications_.size() >= maxCapacity_)
    {
        notifications_.pop();
    }
    notifications_.push(notification);
}

void NotificationHandler::displayNotifications()
{
    std::lock_guard<std::mutex> lock(mutex_);
    size_t count = 0;
    std::cout << "-------------------------------------------------------------------" << std::endl;
    while (!notifications_.empty() && count < 10)
    {
        std::cout << "Notification: " << notifications_.front().dump(4) << std::endl;
        notifications_.pop();
        ++count;
    }
    std::cout << "-------------------------------------------------------------------" << std::endl;


    if (count == 0)
    {
        std::cout << "No notifications available." << std::endl;
        std::cout << "-------------------------------------------------------------------" << std::endl;

    }
}
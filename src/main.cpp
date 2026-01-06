#include "WebSocketHandler.hpp"
#include "MarketAPI.hpp"
#include "Logger.hpp"
#include "NotificationHandler.hpp"

#define MAX_NOTIFS 100

/*
    A personal-use high performance CLI client to trade cryptocurrency using the Deribit Exchange.
    It facilitates secure WebSocket communication, user interaction, and real-time notifications. 

    The codebase is divided into 4 components. 

        1. WebSocket Client: Manages the connection to the Deribit API.

        2. Market API: Handles user commands and constructs API requests.

        3. Notification Handler: Manages a queue of notifications from the API.

        4. Logger: Provides thread-safe logging functionality.

    The Application ensures memory safety by allocation resources at initialization and safely deallocating them and by using scoped locks.

*/


int main() {
    Logger::logInfo("Starting EurekaExchange");

    try {
        // A Notification handler object that takes in the maximum number of notifs to be displayed at once
        NotificationHandler notificationHandler(MAX_NOTIFS);

        WebSocketClient client("test.deribit.com", "443", "/ws/api/v2", notificationHandler);
        client.connect();
        marketAPI::help();
        Logger::logInfo("Welcome to EurekaExchange");
        std::string userInput;

        // User Input Loop that runs in a seperate thread from the websocket connection.

        while (true) {
            sleep(1);
            std::cout << "Enter a message: ";
            std::getline(std::cin, userInput);

            int cleanInput = marketAPI::inputValidator(userInput);
            if (cleanInput == -1) {
                Logger::logWarn("Invalid input received");
                std::cout << "Invalid Input, Please Try Again!" << std::endl;
                std::cout << "-------------------------------------------------------------------" << std::endl;
                continue;
            } else if (cleanInput == 0) {
                Logger::logInfo("Terminating session");
                std::cout << "Thank you for using EurekaExchange" << std::endl;
                std::cout << "-------------------------------------------------------------------" << std::endl;
                exit(0);
            } else if (cleanInput == 1) {
                marketAPI::help();
                continue;
            } else if (cleanInput == 9) {
                notificationHandler.displayNotifications();
                continue;
            }

            std::string message = marketAPI::parse(cleanInput);
            client.sendMessage(message);
        }
    } catch (const std::exception &e) {
        Logger::logError(std::string("Main error: ") + e.what());
        return 1;
    }

    Logger::logInfo("EurekaExchange terminated gracefully");
    return 0;
}


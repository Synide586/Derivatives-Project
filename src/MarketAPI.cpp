#include "MarketAPI.hpp"

using json = nlohmann::json;

namespace marketAPI
{
    // Utility functions to proccess websocket input

    std::string stringInput(const std::string &prompt, const std::regex &pattern, const std::string &error_message)
    {
        std::string input;
        while (true)
        {
            std::cout << prompt;
            std::getline(std::cin, input);
            if (std::regex_match(input, pattern)) {
            return input;
            }
            std::cerr << error_message << std::endl;
        }
    }

    double doubleInput(const std::string &prompt, double min, double max)
    {
        double value;
        while (true)
        {
            std::cout << prompt;
            if (std::cin >> value && value >= min && value <= max)
            {
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                return value;
            }
            std::cerr << "Invalid input. Enter a number between " << min << " and " << max << ".\n";
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
    }

    int inputValidator(const std::string &s)
    {
        if (!s.empty() && std::find_if(s.begin(), s.end(), [](unsigned char c)
                                       { return !std::isdigit(c); }) == s.end())
        {
            int value = std::stoi(s);
            if (value >= 0 && value <= 9)
                return value;
        }
        return -1;
    }

    void help()
    {
        std::cout << "-------------------------------------------------------------------" << std::endl;

        std::cout << "EurekaExchange's functions\n"
                  << "0. Exit Program\n"
                  << "1. Print this help message\n"
                  << "2. Authenticate your identity\n"
                  << "3. Place an order (auth required)\n"
                  << "4. Cancel an order (auth required)\n"
                  << "5. Modify an order (auth required)\n"
                  << "6. Get orderbook (auth required)\n"
                  << "7. View current positions (auth required)\n"
                  << "8. Subscribe to a symbol (auth required)\n"
                  << "9. View unread notifications (auth required)\n";
        std::cout << "-------------------------------------------------------------------" << std::endl;

    }

    std::string parse(int x)
    {
        switch (x)
        {
        case 2:
            return authenticate();
        case 3:
            return buy();
        case 4:
            return cancel();
        case 5:
            return modifyOrder();
        case 6:
            return getOrderbook();
        case 7:
            return getCurrentPositions();
        case 8:
            return subscribe();
        default:
            return R"({"error": "Invalid option"})";
        }
    }

    std::string authenticate()
    {
        std::string client_id = stringInput("Enter your Client ID: ", std::regex("^[a-zA-Z0-9_]+$"), "Invalid Client ID.");
        std::string client_secret = stringInput("Enter your Client Secret: ", std::regex("^[a-zA-Z0-9_]+$"), "Invalid Client Secret.");

        json request = {
            {"jsonrpc", "2.0"},
            {"id", 1},
            {"method", "public/auth"},
            {"params", {{"grant_type", "client_credentials"}, {"client_id", client_id}, {"client_secret", client_secret}}}};

        return request.dump();
    }

    std::string buy()
    {
        std::string instrument = stringInput("Enter the instrument name (e.g., BTC_USDC): ", std::regex("^[A-Za-z0-9_-]+$"), "Invalid instrument name.");
        double amount = doubleInput("Enter the amount to buy: ", 0.00001, 1000000);
        std::string order_type = stringInput("Enter the order type (market/limit): ", std::regex("^(market|limit)$"), "Invalid order type.");

        json request = {
            {"jsonrpc", "2.0"},
            {"id", 2},
            {"method", "private/buy"},
            {"params", {{"instrument_name", instrument}, {"amount", amount}, {"type", order_type}}}};

        return request.dump();
    }

    std::string cancel()
    {
        std::string order_id = stringInput("Enter the Order ID to cancel: ", std::regex("^[A-Za-z0-9_-]+$"), "Invalid Order ID.");

        json request = {
            {"jsonrpc", "2.0"},
            {"id", 3},
            {"method", "private/cancel"},
            {"params", {{"order_id", order_id}}}};

        return request.dump();
    }

    std::string modifyOrder()
    {
        std::string order_id = stringInput("Enter the Order ID to modify: ", std::regex("^[A-Za-z0-9_-]+$"), "Invalid Order ID.");
        double new_amount = doubleInput("Enter the amount to buy: ", 0.00001, 1000000);

        json request = {
            {"jsonrpc", "2.0"},
            {"id", 4},
            {"method", "private/edit"},
            {"params", {{"order_id", order_id}, {"amount", new_amount}}}};

        return request.dump();
    }

    std::string getOrderbook()
    {
        std::string instrument = stringInput("Enter the instrument name (e.g., BTC_USDC): ", std::regex("^[A-Za-z0-9_-]+$"), "Invalid instrument name.");

        json request = {
            {"jsonrpc", "2.0"},
            {"id", 5},
            {"method", "public/get_order_book"},
            {"params", {{"instrument_name", instrument}}}};

        return request.dump();
    }

    std::string getCurrentPositions()
    {
        std::string currency = stringInput("Enter the currency (e.g., BTC_USDC): ", std::regex("^[A-Za-z]+$"), "Invalid currency name.");

        json request = {
            {"jsonrpc", "2.0"},
            {"id", 6},
            {"method", "private/get_positions"},
            {"params", {{"currency", currency}}}};

        return request.dump();
    }

}

std::string marketAPI::subscribe()
{
    std::string symbol = stringInput(
        "Enter the symbol to subscribe to (eg. deribit_price_statistics.ada_usd): ",
        std::regex("^[A-Za-z0-9_.-]+$"),
        "Invalid symbol.");

    json request = {
        {"jsonrpc", "2.0"},
        {"id", 7},
        {"method", "private/subscribe"},
        {"params", {{"channels", json::array({symbol})}}}};
    return request.dump();
}
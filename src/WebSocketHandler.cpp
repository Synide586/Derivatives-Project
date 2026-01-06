#include "WebSocketHandler.hpp"

WebSocketClient::WebSocketClient(const std::string &host, const std::string &port, const std::string &endpoint,
                                 NotificationHandler &notificationHandler)
    : host_(host), port_(port), endpoint_(endpoint), sslCtx_(net::ssl::context::tlsv12_client),
      ws_(ioc_, sslCtx_), notificationHandler_(notificationHandler) {}

WebSocketClient::~WebSocketClient()
{
    if (readThread_.joinable())
    {
        readThread_.join();
    }
}

void WebSocketClient::connect()
{
    try
    {
        tcp::resolver resolver(ioc_);
        auto results = resolver.resolve(host_, port_);
        beast::get_lowest_layer(ws_).connect(results);
        ws_.next_layer().handshake(net::ssl::stream_base::client);
        ws_.handshake(host_, endpoint_);

        Logger::logInfo("WebSocket connection established.");

        readThread_ = std::thread(&WebSocketClient::readLoop, this);
    }
    catch (const std::exception &e)
    {
        Logger::logError(std::string("WebSocket connection error: ") + e.what());
        throw;
    }
}

void WebSocketClient::sendMessage(const std::string &message)
{
    try
    {
        ws_.write(net::buffer(message));
    }
    catch (const std::exception &e)
    {
        Logger::logError(std::string("Send error: ") + e.what());
    }
}

void WebSocketClient::readLoop()
{
    try
    {
        while (true)
        {
            beast::flat_buffer buffer;
            ws_.read(buffer);

            auto response = beast::buffers_to_string(buffer.data());
            nlohmann::json jsonResponse = nlohmann::json::parse(response);

            if (jsonResponse.contains("method") && jsonResponse["method"] == "subscription")
            {
                notificationHandler_.addNotification(jsonResponse);
            }
            else
            {
                Logger::logResult(jsonResponse);
            }
        }
    }
    catch (const std::exception &e)
    {
        Logger::logError(std::string("Read loop error: ") + e.what());
    }
}

void WebSocketClient::close()
{
    try
    {
        ws_.close(websocket::close_code::normal);
        Logger::logInfo("WebSocket closed.");
    }
    catch (const std::exception &e)
    {
        Logger::logError(std::string("Close error: ") + e.what());
    }
}

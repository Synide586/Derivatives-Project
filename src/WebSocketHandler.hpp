#pragma once
#include <boost/beast.hpp>
#include <boost/asio.hpp>
#include <string>
#include <thread>
#include <boost/beast/ssl.hpp>
#include <boost/asio/ssl/stream.hpp>
#include <boost/asio/ip/tcp.hpp>
#include "../include/json.hpp"
#include "NotificationHandler.hpp"
#include "Logger.hpp"

/*
    The WebSocket Client estanblishes a secure connection to the Deribit API.
    It spawns a non blocking thread to communicate with the server via a websocket.

    The thread maintains a read loop of its own which awaits messages from the client, sends the response to the server
    and outputs the response.

    It also handles notifications from subscriptions, and places it in a FIFO queue, which can be accessed by the client.

    Completely Async I/O was not used in order to prevent interference with the read stream and data inaccuracies.

*/
namespace beast = boost::beast;
namespace net = boost::asio;
namespace websocket = beast::websocket;
using tcp = net::ip::tcp;

class WebSocketClient
{
public:
    WebSocketClient(const std::string &host, const std::string &port, const std::string &endpoint,
                    NotificationHandler &notificationHandler);
    ~WebSocketClient();

    void connect();
    void sendMessage(const std::string &message);
    void close();

private:
    void readLoop();

    std::string host_;
    std::string port_;
    std::string endpoint_;
    net::io_context ioc_;
    net::ssl::context sslCtx_;
    websocket::stream<beast::ssl_stream<beast::tcp_stream>> ws_;
    std::thread readThread_;
    NotificationHandler &notificationHandler_;
};
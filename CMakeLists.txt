cmake_minimum_required(VERSION 3.10)
project(EurekaExchange VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

set(SOURCES
    src/main.cpp
    src/WebSocketHandler.cpp
    src/MarketAPI.cpp
    src/Logger.cpp
    src/NotificationHandler.cpp
)

set(HEADERS
    include/json.hpp
    src/Logger.hpp
    src/MarketAPI.hpp
    src/NotificationHandler.hpp
    src/WebSocketHandler.hpp
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME} PRIVATE ssl crypto)

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)

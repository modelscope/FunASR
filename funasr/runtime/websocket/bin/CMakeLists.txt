
add_executable(funasr-wss-server "funasr-wss-server.cpp" "websocket-server.cpp")
add_executable(funasr-wss-server-2pass "funasr-wss-server-2pass.cpp" "websocket-server-2pass.cpp")
add_executable(funasr-wss-client "funasr-wss-client.cpp")
add_executable(funasr-wss-client-2pass "funasr-wss-client-2pass.cpp")

target_link_libraries(funasr-wss-client PUBLIC funasr ssl crypto)
target_link_libraries(funasr-wss-client-2pass PUBLIC funasr ssl crypto)
target_link_libraries(funasr-wss-server PUBLIC funasr ssl crypto)
target_link_libraries(funasr-wss-server-2pass PUBLIC funasr ssl crypto)

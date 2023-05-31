/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
/* 2022-2023 by zhaomingwork */

// client for websocket, support multiple threads
// Usage: websocketclient server_ip port wav_path threads_num

#define ASIO_STANDALONE 1
#include <websocketpp/client.hpp>
#include <websocketpp/common/thread.hpp>
#include <websocketpp/config/asio_client.hpp>

#include "audio.h"
#include "nlohmann/json.hpp"

/**
 * Define a semi-cross platform helper method that waits/sleeps for a bit.
 */
void wait_a_bit() {
#ifdef WIN32
  Sleep(1000);
#else
  sleep(1);
#endif
}
typedef websocketpp::config::asio_client::message_type::ptr message_ptr;
typedef websocketpp::lib::shared_ptr<websocketpp::lib::asio::ssl::context>
    context_ptr;
using websocketpp::lib::bind;
using websocketpp::lib::placeholders::_1;
using websocketpp::lib::placeholders::_2;
context_ptr on_tls_init(websocketpp::connection_hdl) {
  context_ptr ctx = websocketpp::lib::make_shared<asio::ssl::context>(
      asio::ssl::context::sslv23);

  try {
    ctx->set_options(
        asio::ssl::context::default_workarounds | asio::ssl::context::no_sslv2 |
        asio::ssl::context::no_sslv3 | asio::ssl::context::single_dh_use);

  } catch (std::exception& e) {
    std::cout << e.what() << std::endl;
  }
  return ctx;
}
// template for tls or not config
template <typename T>
class websocket_client {
 public:
  // typedef websocketpp::client<T> client;
  // typedef websocketpp::client<websocketpp::config::asio_tls_client>
  // wss_client;
  typedef websocketpp::lib::lock_guard<websocketpp::lib::mutex> scoped_lock;

  websocket_client(int is_ssl) : m_open(false), m_done(false) {
    // set up access channels to only log interesting things

    m_client.clear_access_channels(websocketpp::log::alevel::all);
    m_client.set_access_channels(websocketpp::log::alevel::connect);
    m_client.set_access_channels(websocketpp::log::alevel::disconnect);
    m_client.set_access_channels(websocketpp::log::alevel::app);

    // Initialize the Asio transport policy
    m_client.init_asio();

    // Bind the handlers we are using
    using websocketpp::lib::bind;
    using websocketpp::lib::placeholders::_1;
    m_client.set_open_handler(bind(&websocket_client::on_open, this, _1));
    m_client.set_close_handler(bind(&websocket_client::on_close, this, _1));
    m_client.set_close_handler(bind(&websocket_client::on_close, this, _1));

    m_client.set_message_handler(
        [this](websocketpp::connection_hdl hdl, message_ptr msg) {
          on_message(hdl, msg);
        });

    m_client.set_fail_handler(bind(&websocket_client::on_fail, this, _1));
    m_client.clear_access_channels(websocketpp::log::alevel::all);
  }
  void on_message(websocketpp::connection_hdl hdl, message_ptr msg) {
    const std::string& payload = msg->get_payload();
    switch (msg->get_opcode()) {
      case websocketpp::frame::opcode::text:
        std::cout << "on_message=" << payload << std::endl;
    }
  }
  // This method will block until the connection is complete

  void run(const std::string& uri, const std::string& wav_path) {
    // Create a new connection to the given URI
    websocketpp::lib::error_code ec;
    typename websocketpp::client<T>::connection_ptr con =
        m_client.get_connection(uri, ec);
    if (ec) {
      m_client.get_alog().write(websocketpp::log::alevel::app,
                                "Get Connection Error: " + ec.message());
      return;
    }
    this->wav_path = std::move(wav_path);
    // Grab a handle for this connection so we can talk to it in a thread
    // safe manor after the event loop starts.
    m_hdl = con->get_handle();

    // Queue the connection. No DNS queries or network connections will be
    // made until the io_service event loop is run.
    m_client.connect(con);

    // Create a thread to run the ASIO io_service event loop
    websocketpp::lib::thread asio_thread(&websocketpp::client<T>::run,
                                         &m_client);

    send_wav_data();
    asio_thread.join();
  }

  // The open handler will signal that we are ready to start sending data
  void on_open(websocketpp::connection_hdl) {
    m_client.get_alog().write(websocketpp::log::alevel::app,
                              "Connection opened, starting data!");

    scoped_lock guard(m_lock);
    m_open = true;
  }

  // The close handler will signal that we should stop sending data
  void on_close(websocketpp::connection_hdl) {
    m_client.get_alog().write(websocketpp::log::alevel::app,
                              "Connection closed, stopping data!");

    scoped_lock guard(m_lock);
    m_done = true;
  }

  // The fail handler will signal that we should stop sending data
  void on_fail(websocketpp::connection_hdl) {
    m_client.get_alog().write(websocketpp::log::alevel::app,
                              "Connection failed, stopping data!");

    scoped_lock guard(m_lock);
    m_done = true;
  }
  // send wav to server
  void send_wav_data() {
    uint64_t count = 0;
    std::stringstream val;

    funasr::Audio audio(1);
    int32_t sampling_rate = 16000;

    if (!audio.LoadPcmwav(wav_path.c_str(), &sampling_rate)) {
      std::cout << "error in load wav" << std::endl;
      return;
    }

    float* buff;
    int len;
    int flag = 0;
    bool wait = false;
    while (1) {
      {
        scoped_lock guard(m_lock);
        // If the connection has been closed, stop generating data
        if (m_done) {
          break;
        }

        // If the connection hasn't been opened yet wait a bit and retry
        if (!m_open) {
          wait = true;
        } else {
          break;
        }
      }

      if (wait) {
        std::cout << "wait.." << m_open << std::endl;
        wait_a_bit();

        continue;
      }
    }
    websocketpp::lib::error_code ec;

    nlohmann::json jsonbegin;
    nlohmann::json chunk_size = nlohmann::json::array();
    chunk_size.push_back(5);
    chunk_size.push_back(0);
    chunk_size.push_back(5);
    jsonbegin["chunk_size"] = chunk_size;
    jsonbegin["chunk_interval"] = 10;
    jsonbegin["wav_name"] = "damo";
    jsonbegin["is_speaking"] = true;
    m_client.send(m_hdl, jsonbegin.dump(), websocketpp::frame::opcode::text,
                  ec);

    // fetch wav data use asr engine api
    while (audio.Fetch(buff, len, flag) > 0) {
      short iArray[len];

      // convert float -1,1 to short -32768,32767
      for (size_t i = 0; i < len; ++i) {
        iArray[i] = (short)(buff[i] * 32767);
      }
      // send data to server
      m_client.send(m_hdl, iArray, len * sizeof(short),
                    websocketpp::frame::opcode::binary, ec);
      std::cout << "sended data len=" << len * sizeof(short) << std::endl;
      // The most likely error that we will get is that the connection is
      // not in the right state. Usually this means we tried to send a
      // message to a connection that was closed or in the process of
      // closing. While many errors here can be easily recovered from,
      // in this simple example, we'll stop the data loop.
      if (ec) {
        m_client.get_alog().write(websocketpp::log::alevel::app,
                                  "Send Error: " + ec.message());
        break;
      }

      wait_a_bit();
    }
    nlohmann::json jsonresult;
    jsonresult["is_speaking"] = false;
    m_client.send(m_hdl, jsonresult.dump(), websocketpp::frame::opcode::text,
                  ec);
    wait_a_bit();
  }
  websocketpp::client<T> m_client;

 private:
  websocketpp::connection_hdl m_hdl;
  websocketpp::lib::mutex m_lock;
  std::string wav_path;
  bool m_open;
  bool m_done;
};

int main(int argc, char* argv[]) {
  if (argc < 6) {
    printf("Usage: %s server_ip port wav_path threads_num is_ssl\n", argv[0]);
    exit(-1);
  }
  std::string server_ip = argv[1];
  std::string port = argv[2];
  std::string wav_path = argv[3];
  int threads_num = atoi(argv[4]);
  int is_ssl = atoi(argv[5]);
  std::vector<websocketpp::lib::thread> client_threads;
  std::string uri = "";
  if (is_ssl == 1) {
    uri = "wss://" + server_ip + ":" + port;
  } else {
    uri = "ws://" + server_ip + ":" + port;
  }

  for (size_t i = 0; i < threads_num; i++) {
    client_threads.emplace_back([uri, wav_path, is_ssl]() {
      if (is_ssl == 1) {
        websocket_client<websocketpp::config::asio_tls_client> c(is_ssl);

        c.m_client.set_tls_init_handler(bind(&on_tls_init, ::_1));

        c.run(uri, wav_path);
      } else {
        websocket_client<websocketpp::config::asio_client> c(is_ssl);

        c.run(uri, wav_path);
      }
    });
  }

  for (auto& t : client_threads) {
    t.join();
  }
}
/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
/* 2022-2023 by zhaomingwork */

// client for websocket, support multiple threads
// ./funasr-wss-client  --server-ip <string>
//                     --port <string>
//                     --wav-path <string>
//                     [--thread-num <int>] 
//                     [--is-ssl <int>]  [--]
//                     [--version] [-h]
// example:
// ./funasr-wss-client --server-ip 127.0.0.1 --port 10095 --wav-path test.wav --thread-num 1 --is-ssl 1

#define ASIO_STANDALONE 1
#include <websocketpp/client.hpp>
#include <websocketpp/common/thread.hpp>
#include <websocketpp/config/asio_client.hpp>
#include <fstream>
#include <atomic>
#include <thread>
#include <glog/logging.h>
#include "util.h"
#include "audio.h"
#include "nlohmann/json.hpp"
#include "tclap/CmdLine.h"

/**
 * Define a semi-cross platform helper method that waits/sleeps for a bit.
 */
void WaitABit() {
    #ifdef WIN32
        Sleep(200);
    #else
        usleep(200);
    #endif
}
std::atomic<int> wav_index(0);

typedef websocketpp::config::asio_client::message_type::ptr message_ptr;
typedef websocketpp::lib::shared_ptr<websocketpp::lib::asio::ssl::context> context_ptr;
using websocketpp::lib::bind;
using websocketpp::lib::placeholders::_1;
using websocketpp::lib::placeholders::_2;
context_ptr OnTlsInit(websocketpp::connection_hdl) {
    context_ptr ctx = websocketpp::lib::make_shared<asio::ssl::context>(
        asio::ssl::context::sslv23);

    try {
        ctx->set_options(
            asio::ssl::context::default_workarounds | asio::ssl::context::no_sslv2 |
            asio::ssl::context::no_sslv3 | asio::ssl::context::single_dh_use);

    } catch (std::exception& e) {
        LOG(ERROR) << e.what();
    }
    return ctx;
}

// template for tls or not config
template <typename T>
class WebsocketClient {
  public:
    // typedef websocketpp::client<T> client;
    // typedef websocketpp::client<websocketpp::config::asio_tls_client>
    // wss_client;
    typedef websocketpp::lib::lock_guard<websocketpp::lib::mutex> scoped_lock;

    WebsocketClient(int is_ssl) : m_open(false), m_done(false) {
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
        m_client.set_open_handler(bind(&WebsocketClient::on_open, this, _1));
        m_client.set_close_handler(bind(&WebsocketClient::on_close, this, _1));

        m_client.set_message_handler(
            [this](websocketpp::connection_hdl hdl, message_ptr msg) {
              on_message(hdl, msg);
            });

        m_client.set_fail_handler(bind(&WebsocketClient::on_fail, this, _1));
        m_client.clear_access_channels(websocketpp::log::alevel::all);
    }

    void on_message(websocketpp::connection_hdl hdl, message_ptr msg) {
        const std::string& payload = msg->get_payload();
        switch (msg->get_opcode()) {
            case websocketpp::frame::opcode::text:
				total_recv=total_recv+1;
                LOG(INFO)<< "Thread: " << this_thread::get_id() << ", total_recv=" << total_recv <<", on_message = " << payload;
                std::unique_lock<std::mutex> lock(msg_lock);
                cv.notify_one();
				if(close_client)
				{
                    LOG(INFO)<< "Thread: " << this_thread::get_id() << ", close client thread";
					websocketpp::lib::error_code ec;
					m_client.close(m_hdl, websocketpp::close::status::going_away, "", ec);
					if (ec){
                        LOG(ERROR)<< "Error closing connection " << ec.message();
					}
				}
        }
    }

    // This method will block until the connection is complete  
    void run(const std::string& uri, const std::vector<string>& wav_list, const std::vector<string>& wav_ids, 
             int audio_fs, const std::unordered_map<std::string, int>& hws_map, int use_itn=1, int svs_itn=1) {
        // Create a new connection to the given URI
        websocketpp::lib::error_code ec;
        typename websocketpp::client<T>::connection_ptr con =
            m_client.get_connection(uri, ec);
        if (ec) {
            m_client.get_alog().write(websocketpp::log::alevel::app,
                                    "Get Connection Error: " + ec.message());
            return;
        }
        // Grab a handle for this connection so we can talk to it in a thread
        // safe manor after the event loop starts.
        m_hdl = con->get_handle();

        // Queue the connection. No DNS queries or network connections will be
        // made until the io_service event loop is run.
        m_client.connect(con);

        // Create a thread to run the ASIO io_service event loop
        websocketpp::lib::thread asio_thread(&websocketpp::client<T>::run,
                                            &m_client);
        bool send_hotword = true;
        while(true){
            int i = wav_index.fetch_add(1);
            if (i >= wav_list.size()) {
                break;
            }
            if (total_send !=0){
                std::unique_lock<std::mutex> lock(msg_lock);
                cv.wait(lock);
            }
            total_send += 1;
            send_wav_data(wav_list[i], wav_ids[i], audio_fs, hws_map, send_hotword, use_itn, svs_itn);
            if(send_hotword){
                send_hotword = false;
            }
        }
        close_client = true;
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
    void send_wav_data(string wav_path, string wav_id, int audio_fs,
        const std::unordered_map<std::string, int>& hws_map, 
        bool send_hotword, bool use_itn, bool svs_itn) {
        uint64_t count = 0;
        std::stringstream val;

		funasr::Audio audio(1);
        int32_t sampling_rate = audio_fs;
        std::string wav_format = "pcm";
        if(funasr::IsTargetFile(wav_path.c_str(), "pcm")){
			if (!audio.LoadPcmwav(wav_path.c_str(), &sampling_rate, false))
				return ;
		}else{
			wav_format = "others";
            if (!audio.LoadOthers2Char(wav_path.c_str()))
				return ;
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
                // LOG(INFO) << "wait.." << m_open;
                WaitABit();
                continue;
            }
        }
        websocketpp::lib::error_code ec;

        nlohmann::json jsonbegin;
        nlohmann::json chunk_size = nlohmann::json::array();
        chunk_size.push_back(5);
        chunk_size.push_back(10);
        chunk_size.push_back(5);
        jsonbegin["chunk_size"] = chunk_size;
        jsonbegin["chunk_interval"] = 10;
        jsonbegin["wav_name"] = wav_id;
        jsonbegin["wav_format"] = wav_format;
        jsonbegin["audio_fs"] = sampling_rate;
        jsonbegin["itn"] = true;
        jsonbegin["svs_itn"] = true;
        if(use_itn == 0){
            jsonbegin["itn"] = false;
        }
        if(svs_itn == 0){
            jsonbegin["svs_itn"] = false;
        }
        jsonbegin["is_speaking"] = true;
        if(send_hotword){
            if(!hws_map.empty()){
                LOG(INFO) << "hotwords: ";
                for (const auto& pair : hws_map) {
                    LOG(INFO) << pair.first << " : " << pair.second;
                }
                nlohmann::json json_map(hws_map);
                std::string json_map_str = json_map.dump();
                jsonbegin["hotwords"] = json_map_str;
            }
        }
        m_client.send(m_hdl, jsonbegin.dump(), websocketpp::frame::opcode::text,
                      ec);

        // fetch wav data use asr engine api
        if(wav_format == "pcm"){
            while (audio.Fetch(buff, len, flag) > 0) {
                short* iArray = new short[len];
                for (size_t i = 0; i < len; ++i) {
                iArray[i] = (short)(buff[i]*32768);
                }

                // send data to server
                int offset = 0;
                int block_size = 102400;
                while(offset < len){
                    int send_block = 0;
                    if (offset + block_size <= len){
                        send_block = block_size;
                    }else{
                        send_block = len - offset;
                    }
                    m_client.send(m_hdl, iArray+offset, send_block * sizeof(short),
                        websocketpp::frame::opcode::binary, ec);
                    offset += send_block;
                }

                LOG(INFO)<< "Thread: " << this_thread::get_id() << ", sended data len=" << len * sizeof(short);
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
                delete[] iArray;
                // WaitABit();
            }
        }else{
            int offset = 0;
            int block_size = 204800;
            len = audio.GetSpeechLen();
            char* others_buff = audio.GetSpeechChar();

            while(offset < len){
                int send_block = 0;
                if (offset + block_size <= len){
                    send_block = block_size;
                }else{
                    send_block = len - offset;
                }
                m_client.send(m_hdl, others_buff+offset, send_block,
                    websocketpp::frame::opcode::binary, ec);
                offset += send_block;
            }

            LOG(INFO)<< "Thread: " << this_thread::get_id() << ", sended data len=" << len;
            // The most likely error that we will get is that the connection is
            // not in the right state. Usually this means we tried to send a
            // message to a connection that was closed or in the process of
            // closing. While many errors here can be easily recovered from,
            // in this simple example, we'll stop the data loop.
            if (ec) {
                m_client.get_alog().write(websocketpp::log::alevel::app,
                                        "Send Error: " + ec.message());
            }
        }

        nlohmann::json jsonresult;
        jsonresult["is_speaking"] = false;
        m_client.send(m_hdl, jsonresult.dump(), websocketpp::frame::opcode::text,
                      ec);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    websocketpp::client<T> m_client;

  private:
    websocketpp::connection_hdl m_hdl;
    websocketpp::lib::mutex m_lock;
    websocketpp::lib::mutex msg_lock;
    websocketpp::lib::condition_variable cv;
    bool m_open;
    bool m_done;
    bool close_client=false;
	int total_send=0;
    int total_recv=0;
};

int main(int argc, char* argv[]) {
#ifdef _WIN32
    #include <windows.h>
    SetConsoleOutputCP(65001);
#endif
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;

    TCLAP::CmdLine cmd("funasr-wss-client", ' ', "1.0");
    TCLAP::ValueArg<std::string> server_ip_("", "server-ip", "server-ip", true,
                                           "127.0.0.1", "string");
    TCLAP::ValueArg<std::string> port_("", "port", "port", true, "10095", "string");
    TCLAP::ValueArg<std::string> wav_path_("", "wav-path", 
        "the input could be: wav_path, e.g.: asr_example.wav; pcm_path, e.g.: asr_example.pcm; wav.scp, kaldi style wav list (wav_id \t wav_path)", 
        true, "", "string");
    TCLAP::ValueArg<std::int32_t> audio_fs_("", "audio-fs", "the sample rate of audio", false, 16000, "int32_t");
    TCLAP::ValueArg<int> thread_num_("", "thread-num", "thread-num",
                                       false, 1, "int");
    TCLAP::ValueArg<int> is_ssl_(
        "", "is-ssl", "is-ssl is 1 means use wss connection, or use ws connection", 
        false, 1, "int");
    TCLAP::ValueArg<int> use_itn_(
        "", "use-itn",
        "use-itn is 1 means use itn, 0 means not use itn", false, 1, "int");
    TCLAP::ValueArg<int> svs_itn_(
        "", "svs-itn",
        "svs-itn is 1 means use itn and punc, 0 means not use", false, 1, "int");
    TCLAP::ValueArg<std::string> hotword_("", HOTWORD,
        "the hotword file, one hotword perline, Format: Hotword Weight (could be: 阿里巴巴 20)", false, "", "string");

    cmd.add(server_ip_);
    cmd.add(port_);
    cmd.add(wav_path_);
    cmd.add(audio_fs_);
    cmd.add(thread_num_);
    cmd.add(is_ssl_);
    cmd.add(use_itn_);
    cmd.add(svs_itn_);
    cmd.add(hotword_);
    cmd.parse(argc, argv);

    std::string server_ip = server_ip_.getValue();
    std::string port = port_.getValue();
    std::string wav_path = wav_path_.getValue();
    int threads_num = thread_num_.getValue();
    int is_ssl = is_ssl_.getValue();
    int use_itn = use_itn_.getValue();
    int svs_itn = svs_itn_.getValue();

    std::vector<websocketpp::lib::thread> client_threads;
    std::string uri = "";
    if (is_ssl == 1) {
        uri = "wss://" + server_ip + ":" + port;
    } else {
        uri = "ws://" + server_ip + ":" + port;
    }

    // hotwords
    std::string hotword_path = hotword_.getValue();
    unordered_map<string, int> hws_map;
    if(!hotword_path.empty()){
        LOG(INFO) << "hotword path: " << hotword_path;
        funasr::ExtractHws(hotword_path, hws_map);
    }

    // read wav_path
    std::vector<string> wav_list;
    std::vector<string> wav_ids;
    string default_id = "wav_default_id";
    if(funasr::IsTargetFile(wav_path, "scp")){
        ifstream in(wav_path);
        if (!in.is_open()) {
            printf("Failed to open scp file");
            return 0;
        }
        string line;
        while(getline(in, line))
        {
            istringstream iss(line);
            string column1, column2;
            iss >> column1 >> column2;
            wav_list.emplace_back(column2);
            wav_ids.emplace_back(column1);
        }
        in.close();
    }else{
        wav_list.emplace_back(wav_path);
        wav_ids.emplace_back(default_id);
    }
    
    int audio_fs = audio_fs_.getValue();
    for (size_t i = 0; i < threads_num; i++) {
        client_threads.emplace_back([uri, wav_list, wav_ids, audio_fs, is_ssl, hws_map, use_itn, svs_itn]() {
          if (is_ssl == 1) {
            WebsocketClient<websocketpp::config::asio_tls_client> c(is_ssl);

            c.m_client.set_tls_init_handler(bind(&OnTlsInit, ::_1));

            c.run(uri, wav_list, wav_ids, audio_fs, hws_map, use_itn, svs_itn);
          } else {
            WebsocketClient<websocketpp::config::asio_client> c(is_ssl);

            c.run(uri, wav_list, wav_ids, audio_fs, hws_map, use_itn, svs_itn);
          }
        });
    }

    for (auto& t : client_threads) {
        t.join();
    }
}

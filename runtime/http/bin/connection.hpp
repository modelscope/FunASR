/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
/* 2023-2025 by zhaomingwork@qq.com */
//
// copy some codes from  http://www.boost.org/
//

#ifndef HTTP_SERVER2_CONNECTION_HPP
#define HTTP_SERVER2_CONNECTION_HPP

#include <array>
#include <asio.hpp>
#include <atomic>
#include <iostream>
#include <memory>

#include "reply.hpp"

#include <fstream>

#include <boost/beast.hpp>

#include "model-decoder.h"

namespace beast = boost::beast;
namespace beasthttp = beast::http;

extern std::unordered_map<std::string, int> hws_map_;
extern int fst_inc_wts_;
extern float global_beam_, lattice_beam_, am_scale_;

namespace http
{
    namespace server2
    {

        /// Represents a single connection from a client.
        class connection : public std::enable_shared_from_this<connection>
        {
        public:
            connection(const connection &) = delete;
            connection &operator=(const connection &) = delete;
            ~connection()
            {
                std::cout << "one connection is close()" << std::endl;
            };

            /// Construct a connection with the given socket.
            explicit connection(asio::ip::tcp::socket socket,
                                asio::io_context &io_decoder, int connection_id,
                                std::shared_ptr<ModelDecoder> model_decoder);

            /// Start the first asynchronous operation for the connection.
            void start();
            std::shared_ptr<FUNASR_MESSAGE> &get_data_msg();
            void write_back(std::string str);

            // 处理100 Continue逻辑
            void handle_100_continue()
            {
                //  start_timer(5); // 5秒超时

                auto self = shared_from_this();
                const std::string response =
                    "HTTP/1.1 100 Continue\r\n"
                    "Connection: keep-alive\r\n\r\n";

                asio::async_write(socket_, asio::buffer(response),
                                  [this, self](asio::error_code ec, size_t)
                                  {
                                      if (ec)
                                          return handle_error(ec);

                                      state_ = State::ReadingHeaders;

                                      do_read();
                                  });
            }

            // 准备文件存储
            void prepare_body_handling()
            {
                if (!filename_.empty())
                {
                    sanitize_filename(filename_);
                    output_file_.open(filename_, std::ios::binary);
                    if (!output_file_)
                    {
                        std::cerr << "Failed to open: " << filename_ << "\n";
                        socket_.close();
                    }
                }
            }

            void finalize_request()
            {
                std::cout << "finalize_request" << std::endl;
                send_final_response();
            }

            void send_final_response()
            {
                const std::string response =
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Length: 0\r\n\r\n";
                asio::write(socket_, asio::buffer(response));
                socket_.close();
            }

            void send_417_expectation_failed()
            {
                const std::string response =
                    "HTTP/1.1 417 Expectation Failed\r\n"
                    "Connection: close\r\n\r\n";
                asio::write(socket_, asio::buffer(response));
                socket_.close();
            }

            // 安全处理文件名
            static void sanitize_filename(std::string &name)
            {
                std::replace(name.begin(), name.end(), '/', '_');
                std::replace(name.begin(), name.end(), '\\', '_');
                name = name.substr(name.find_last_of(":") + 1); // 移除潜在路径
            }

            // 协议版本解析
            bool parse_http_version(const std::string &headers)
            {
                size_t start = headers.find("HTTP/");
                if (start == std::string::npos)
                    return false;

                start += 5;
                size_t dot = headers.find('.', start);
                if (dot == std::string::npos)
                    return false;

                try
                {
                    http_version_major_ = std::stoi(headers.substr(start, dot - start));
                    http_version_minor_ = std::stoi(headers.substr(dot + 1, 1));
                    return true;
                }
                catch (...)
                {
                    return false;
                }
            }
            // 头部解析
            bool try_parse_headers()
            {

                size_t header_end = received_data_.find("\r\n\r\n");
                if (header_end == std::string::npos)
                {

                    return false;
                }

                std::string headers = received_data_.substr(0, header_end);

                //  解析内容信息
                if (content_length_ <= 0)
                    content_length_ = parse_content_length(headers);
                // 解析HTTP版本
                if (!parse_http_version(headers))
                {
                    return false;
                }

                // 检查Expect头
                std::string continue100 = "Expect: 100-continue";
                size_t pos = headers.find(continue100);
                expect_100_continue_ = pos != std::string::npos;

                // 检查协议兼容性
                if (expect_100_continue_)
                {

                    headers.erase(pos, continue100.length());

                    received_data_ = headers;
                    state_ = State::SendingContinue;
                    if (http_version_minor_ < 1)
                        send_417_expectation_failed();
                    return true;
                }

                filename_ = parse_attachment_filename(headers);

                // 状态转移
                std::string ext = parese_file_ext(filename_);

                if (filename_.find(".wav") != std::string::npos)
                {
                    std::cout << "set wav_format=pcm, file_name=" << filename_
                              << std::endl;
                    data_msg->msg["wav_format"] = "pcm";
                }
                else
                {
                    std::cout << "set wav_format=" << ext << ", file_name=" << filename_
                              << std::endl;
                    data_msg->msg["wav_format"] = ext;
                }
                data_msg->msg["wav_name"] = filename_;
 
                state_ = State::ReadingBody;
                return true;
            }

            void parse_multipart_boundary()
            {
                size_t content_type_pos = received_data_.find("Content-Type: multipart/form-data");
                if (content_type_pos == std::string::npos)
                    return;

                size_t boundary_pos = received_data_.find("boundary=", content_type_pos);
                if (boundary_pos == std::string::npos)
                    return;

                boundary_pos += 9; // "boundary="长度
                size_t boundary_end = received_data_.find("\r\n", boundary_pos);
                boundary_ = received_data_.substr(boundary_pos, boundary_end - boundary_pos);

                // 清理boundary的引号
                if (boundary_.front() == '"' && boundary_.back() == '"')
                {
                    boundary_ = boundary_.substr(1, boundary_.size() - 2);
                }
            }
            // multipart 数据处理核心
            void process_multipart_data()
            {
                if (boundary_.empty())
                {
                    parse_multipart_boundary();
                    if (boundary_.empty())
                    {
                        std::cerr << "Invalid multipart format\n";
                        return;
                    }
                }

                while (true)
                {
                    if (!in_file_part_)
                    {

                        // 查找boundary起始
                        size_t boundary_pos = received_data_.find("--" + boundary_);
                        if (boundary_pos == std::string::npos)
                            break;

                        // 移动到part头部
                        size_t part_start = received_data_.find("\r\n\r\n", boundary_pos);
                        if (part_start == std::string::npos)
                            break;

                        part_start += 4; // 跳过空行
                        parse_part_headers(received_data_.substr(boundary_pos, part_start - boundary_pos));

                        received_data_.erase(0, part_start);

                        in_file_part_ = true;
                    }
                    else
                    {
                        // 查找boundary结束

                        size_t boundary_end = received_data_.find("\r\n--" + boundary_);

                        if (boundary_end == std::string::npos)

                            break;

                        // 写入内容

                        std::string tmpstr = received_data_.substr(0, boundary_end);
                        data_msg->samples->insert(data_msg->samples->end(), tmpstr.begin(), tmpstr.end());

                        received_data_.erase(0, boundary_end + 2); // 保留\r\n供下次解析

                        in_file_part_ = false;
                    }
                }
            }
            std::string parese_file_ext(std::string file_name)
            {
                int pos = file_name.rfind('.');
                std::string ext = "";
                if (pos != std::string::npos)
                    ext = file_name.substr(pos + 1);

                return ext;
            }
            // 解析part头部信息
            void parse_part_headers(const std::string &headers)
            {
                current_part_filename_.clear();
                expected_part_size_ = 0;

                // 解析文件名
                size_t filename_pos = headers.find("filename=\"");
                if (filename_pos != std::string::npos)
                {
                    filename_pos += 10;
                    size_t filename_end = headers.find('"', filename_pos);
                    current_part_filename_ = headers.substr(filename_pos, filename_end - filename_pos);
                    sanitize_filename(current_part_filename_);
                }

                // 解析Content-Length
                size_t cl_pos = headers.find("Content-Length: ");
                if (cl_pos != std::string::npos)
                {
                    cl_pos += 15;
                    size_t cl_end = headers.find("\r\n", cl_pos);
                    expected_part_size_ = std::stoull(headers.substr(cl_pos, cl_end - cl_pos));
                }
            }

        private:
            /// Perform an asynchronous read operation.
            void do_read();
            void handle_body();
            std::string parse_attachment_filename(const std::string &header);
            size_t parse_content_length(const std::string &header);
        
            void handle_error(asio::error_code ec);
            /// Perform an asynchronous write operation.
            void do_write();

            void do_decoder();

            void setup_timer();

            /// Socket for the connection.
            asio::ip::tcp::socket socket_;

            /// Buffer for incoming data.
            std::array<char, 8192> buffer_;
            /// for time out
            std::shared_ptr<asio::steady_timer> s_timer;

            std::shared_ptr<ModelDecoder> model_decoder;

            int connection_id = 0;

            /// The reply to be sent back to the client.
            reply reply_;

            asio::io_context &io_decoder;

            std::shared_ptr<FUNASR_MESSAGE> data_msg;

            std::mutex m_lock;

            std::shared_ptr<asio::io_context::strand> strand_;

 

            beasthttp::response_parser<beasthttp::string_body> parser_; // 渐进式解析器

            std::string received_data_;  // 累积接收的数据
            bool header_parsed_ = false; // 头部解析状态标记
            size_t content_length_ = 0;  // Content-Length 值
            enum class State
            {
                ReadingHeaders,
                SendingContinue,
                ReadingBody
            };
            bool expect_100_continue_ = false;
            State state_ = State::ReadingHeaders;
            std::string filename_;
            std::ofstream output_file_;
            int http_version_major_ = 1;
            int http_version_minor_ = 1;
            std::string boundary_ = "";
            bool in_file_part_ = false;
            std::string current_part_filename_;
            size_t expected_part_size_ = 0;
        };

        typedef std::shared_ptr<connection> connection_ptr;

    } // namespace server2
} // namespace http

#endif // HTTP_SERVER2_CONNECTION_HPP

/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
/* 2023-2025 by zhaomingwork@qq.com */
//
// connection.cpp
// copy some codes from  http://www.boost.org/
#include "connection.hpp"

#include <thread>
#include <utility>
#include "util.hpp"


namespace http
{
  namespace server2
  {
     
    std::shared_ptr<FUNASR_MESSAGE> &connection::get_data_msg() { return data_msg; }
    connection::connection(asio::ip::tcp::socket socket,
                           asio::io_context &io_decoder, int connection_id,
                           std::shared_ptr<ModelDecoder> model_decoder)
        : socket_(std::move(socket)),
          io_decoder(io_decoder),
          connection_id(connection_id),
          model_decoder(model_decoder)

    {
      s_timer = std::make_shared<asio::steady_timer>(io_decoder);
    }

    void connection::setup_timer()
    {
      if (data_msg->status == 1)
        return;

      s_timer->expires_after(std::chrono::seconds(10));
      s_timer->async_wait([=](const asio::error_code &ec)
                          {
    if (!ec) {
      std::cout << "time is out!" << std::endl;
      if (data_msg->status == 1) return;
      data_msg->status = 1;
      s_timer->cancel();
      auto wf = std::bind(&connection::write_back, std::ref(*this), "");
      // close the connection
      strand_->post(wf);
    } });
    }

    void connection::start()
    {
      std::lock_guard<std::mutex> lock(m_lock); // for threads safty
      try
      {

        data_msg = std::make_shared<FUNASR_MESSAGE>(); // put a new data vector for
                                                       // new connection
        data_msg->samples = std::make_shared<std::vector<char>>();
        // data_msg->samples->reserve(16000*20);
        data_msg->msg = nlohmann::json::parse("{}");
        data_msg->msg["wav_format"] = "pcm";
        data_msg->msg["wav_name"] = "wav-default-id";
        data_msg->msg["itn"] = true;
        data_msg->msg["audio_fs"] = 16000; // default is 16k
        data_msg->msg["access_num"] = 0;   // the number of access for this object,
                                           // when it is 0, we can free it saftly
        data_msg->msg["is_eof"] = false;
        data_msg->status = 0;

        strand_ = std::make_shared<asio::io_context::strand>(io_decoder);

        FUNASR_DEC_HANDLE decoder_handle = FunASRWfstDecoderInit(
            model_decoder->get_asr_handle(), ASR_OFFLINE, global_beam_, lattice_beam_, am_scale_);

        data_msg->decoder_handle = decoder_handle;

        if (data_msg->hotwords_embedding == nullptr)
        {
          std::unordered_map<std::string, int> merged_hws_map;
          std::string nn_hotwords = "";

          if (true)
          {
            std::string json_string = "{}";
            if (!json_string.empty())
            {
              nlohmann::json json_fst_hws;
              try
              {
                json_fst_hws = nlohmann::json::parse(json_string);
                if (json_fst_hws.type() == nlohmann::json::value_t::object)
                {
                  // fst
                  try
                  {
                    std::unordered_map<std::string, int> client_hws_map =
                        json_fst_hws;
                    merged_hws_map.insert(client_hws_map.begin(),
                                          client_hws_map.end());
                  }
                  catch (const std::exception &e)
                  {
                    std::cout << e.what();
                  }
                }
              }
              catch (std::exception const &e)
              {
                std::cout << e.what();
                // nn
                std::string client_nn_hws = "{}";
                nn_hotwords += " " + client_nn_hws;
                std::cout << "nn hotwords: " << client_nn_hws;
              }
            }
          }
          merged_hws_map.insert(hws_map_.begin(), hws_map_.end());

          // fst
          std::cout << "hotwords: ";
          for (const auto &pair : merged_hws_map)
          {
            nn_hotwords += " " + pair.first;
            std::cout << pair.first << " : " << pair.second;
          }
          FunWfstDecoderLoadHwsRes(data_msg->decoder_handle, fst_inc_wts_,
                                   merged_hws_map);

          // nn
          std::vector<std::vector<float>> new_hotwords_embedding =
              CompileHotwordEmbedding(model_decoder->get_asr_handle(), nn_hotwords);
          data_msg->hotwords_embedding =
              std::make_shared<std::vector<std::vector<float>>>(
                  new_hotwords_embedding);
        }

        
        do_read();
      }
      catch (const std::exception &e)
      {
        std::cout << "error:" << e.what();
      }
    }

    void connection::write_back(std::string str)
    {
      s_timer->cancel();

      reply_ = reply::stock_reply(
          data_msg->msg["asr_result"].dump()); // reply::stock_reply();
      do_write();
    }
    void connection::do_read()
    {

      if (data_msg->status == 1)
        return;

      s_timer->cancel();
      setup_timer();
      auto self(shared_from_this());
      socket_.async_read_some(
          asio::buffer(buffer_),
          [this, self](asio::error_code ec, std::size_t bytes_transferred)
          {
            if (ec)
            {
              handle_error(ec);
              return;
            }

            // 将新数据追加到累积缓冲区
            received_data_.append(buffer_.data(), bytes_transferred);

            switch (state_)
            {
            case State::ReadingHeaders:

              if (try_parse_headers())
              {
                if (state_ == State::SendingContinue)
                {

                  handle_100_continue();
                }
                else
                {

                  handle_body();
                }
              }
              else
              {

                do_read();
              }
              break;

            case State::ReadingBody:
              
              handle_body();
              break;

            case State::SendingContinue:
               
              break; // 等待100 Continue发送完成
            }
          });
    }

    std::string connection::parse_attachment_filename(const std::string &header)
    {
    
      size_t pos = header.find("Content-Disposition: ");
      if (pos == std::string::npos)
        return "";

      pos += 21; // "Content-Disposition: "长度
      size_t end = header.find("\r\n", pos);
      if (end == std::string::npos)
        return "";

      // 调用解析函数
      return parse_attachment_filename_impl(header.substr(pos, end - pos));
    }

    void connection::handle_body()
    {
       

      process_multipart_data();
      if (in_file_part_ == false)
      {
        std::cout << "文件获取结束" << std::endl;
        std::cout << "开始解码，数据大小= " << data_msg->samples->size() << std::endl;

        auto close_thread = std::bind(&connection::write_back, std::ref(*this), "close");
        auto decoder_thread = std::bind(&ModelDecoder::do_decoder,
                           std::ref(*model_decoder), std::ref(data_msg));

        // for decode task
        strand_->post(decoder_thread);
        // for close task
        strand_->post(close_thread);
        data_msg->sem_resultok.acquire();
        std::cout << "解码线程提交结束！！！！ " << std::endl;
      }
      else
        do_read();
      return;
       
    }

    // 辅助函数：解析 Content-Length
    size_t connection::parse_content_length(const std::string &header)
    {
      size_t pos = header.find("Content-Length: ");
      if (pos == std::string::npos)
        return 0;

      pos += 16; // "Content-Length: "长度
      size_t end = header.find("\r\n", pos);
      if (end == std::string::npos)
        return 0;

      try
      {
        return std::stoul(header.substr(pos, end - pos));
      }
      catch (...)
      {
        return 0;
      }
    }

    void connection::handle_error(asio::error_code ec)
    {
      if (ec == asio::error::eof)
      {
        std::cout << "Connection closed gracefully\n";
      }
      else
      {
        std::cerr << "Error: " << ec.message() << "\n";
      }
    }
    void connection::do_write()
    {
      auto self(shared_from_this());
      asio::async_write(socket_, reply_.to_buffers(),
                        [this, self](asio::error_code ec, std::size_t)
                        {
                          if (!ec)
                          {
                            // Initiate graceful connection closure.
                            asio::error_code ignored_ec;
                            socket_.shutdown(asio::ip::tcp::socket::shutdown_both,
                                             ignored_ec);
                          }
                          data_msg->sem_resultok.release();
                          // No new asynchronous operations are started. This means
                          // that all shared_ptr references to the connection object
                          // will disappear and the object will be destroyed
                          // automatically after this handler returns. The
                          // connection class's destructor closes the socket.
                        });
    }

  } // namespace server2
} // namespace http

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <memory>
#include <string>

#include <grpc/grpc.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/security/server_credentials.h>

#include "paraformer.grpc.pb.h"
#include "paraformer_server.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerReaderWriter;
using grpc::ServerWriter;
using grpc::Status;


using paraformer::Request;
using paraformer::Response;
using paraformer::ASR;

ASRServicer::ASRServicer() {
    std::cout << "ASRServicer init" << std::endl;
    init_flag = 0;
}

void ASRServicer::clear_states(const std::string& user) {
    clear_buffers(user);
    clear_transcriptions(user);
}

void ASRServicer::clear_buffers(const std::string& user) {
    if (client_buffers.count(user)) {
        client_buffers.erase(user);
    }
}

void ASRServicer::clear_transcriptions(const std::string& user) {
    if (client_transcription.count(user)) {
        client_transcription.erase(user);
    }
}

void ASRServicer::disconnect(const std::string& user) {
    clear_states(user);
    std::cout << "Disconnecting user: " << user << std::endl;
}

grpc::Status ASRServicer::Recognize(
    grpc::ServerContext* context,
    grpc::ServerReaderWriter<Response, Request>* stream) {

    Request req;
    while (stream->Read(&req)) {
        if (req.isend()) {
            std::cout << "asr end" << std::endl;
            disconnect(req.user());
            Response res;
            res.set_sentence(
                R"({"success": true, "detail": "asr end"})"
            );
            res.set_user(req.user());
            res.set_action("terminate");
            res.set_language(req.language());
            stream->Write(res);
        } else if (req.speaking()) {
            if (req.audio_data().size() > 0) {
                auto& buf = client_buffers[req.user()];
                buf.insert(buf.end(), req.audio_data().begin(), req.audio_data().end());
            }
            Response res;
            res.set_sentence(
                R"({"success": true, "detail": "speaking"})"
            );
            res.set_user(req.user());
            res.set_action("speaking");
            res.set_language(req.language());
            stream->Write(res);
        } else if (!req.speaking()) {
            if (client_buffers.count(req.user()) == 0) {
                Response res;
                res.set_sentence(
                    R"({"success": true, "detail": "waiting_for_voice"})"
                );
                res.set_user(req.user());
                res.set_action("waiting");
                res.set_language(req.language());
                stream->Write(res);
            }else {
                auto begin_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                std::string tmp_data = this->client_buffers[req.user()];
                this->clear_states(req.user());
                
                Response res;
                res.set_sentence(
                    R"({"success": true, "detail": "decoding data: " + std::to_string(tmp_data.length()) + " bytes"})"
                );
                std::string data_len = std::to_string(tmp_data.length());
                std::stringstream ss;
                ss << R"({"success": true, "detail": "decoding data: )" << data_len << R"( bytes")"  << R"("})";
                std::string result = ss.str();
                res.set_sentence(result);
                res.set_user(req.user());
                res.set_action("decoding");
                res.set_language(req.language());
                stream->Write(res);
                if (tmp_data.length() < 800) { //min input_len for asr model
                    auto end_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                    std::string delay_str = std::to_string(end_time - begin_time);
                    std::cout << "user: " << req.user() << " , delay(ms): " << delay_str << ", error: data_is_not_long_enough" << std::endl;
                    Response res;
                    std::stringstream ss;
                    std::string asr_result = "";
                    ss << R"({"success": true, "detail": "finish_sentence","server_delay_ms":)" << delay_str << R"(,"text":")" << asr_result << R"("})";
                    std::string result = ss.str();
                    res.set_sentence(result);
                    res.set_user(req.user());
                    res.set_action("finish");
                    res.set_language(req.language());
                    
                    
                    
                    stream->Write(res);
                }
                else {

		    // asr_result = onnx.infer(tmp_data)
                    /* if (asr_result.find("text") != asr_result.end()) {
                        asr_result = asr_result["text"];
                    }
                    else {
                        asr_result = "";
                    } */

                    std::string asr_result = "你好你好，我是asr识别结果。static";

                    auto end_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                    std::string delay_str = std::to_string(end_time - begin_time);
                    
                    std::cout << "user: " << req.user() << " , delay(ms): " << delay_str << ", text: " << asr_result << std::endl;
                    Response res;
                    std::stringstream ss;
                    ss << R"({"success": true, "detail": "finish_sentence","server_delay_ms":)" << delay_str << R"(,"text":")" << asr_result << R"("})";
                    std::string result = ss.str();
                    res.set_sentence(result);
                    res.set_user(req.user());
                    res.set_action("finish");
                    res.set_language(req.language());
                    
                    
                    stream->Write(res);
                }
            }
        }else {
            Response res;
            res.set_sentence(
                R"({"success": false, "detail": "error, no condition matched! Unknown reason."})"
            );
            res.set_user(req.user());
            res.set_action("terminate");
            res.set_language(req.language());
            stream->Write(res);
        }
    }    
    return Status::OK;
}


void RunServer() {
  std::string server_address("0.0.0.0:10108");
  ASRServicer service;

  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;
  server->Wait();
}

int main(int argc, char** argv) {
 RunServer();
 return 0;
}

/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
 */

#include <fstream>
#include "precomp.h"
#include <vector>

template <typename T>
void print_vec_shape(const std::vector<std::vector<T>> &data)
{
    std::cout << "vec_shape= [" << data.size() << ", ";
    if (!data.empty())
    {
        std::cout << data[0].size();
    }
    std::cout << "]" << std::endl;
}

namespace funasr
{
    void CamPPlusSv::InitSv(const std::string &model, const std::string &cmvn, const std::string &config, int thread_num)
    {
        session_options_.SetIntraOpNumThreads(thread_num);
        session_options_.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
        session_options_.DisableCpuMemArena();

        ReadModel(model.c_str());
        // LoadCmvn(cmvn.c_str());
        LoadConfigFromYaml(config.c_str());
        
    }

    void CamPPlusSv::LoadConfigFromYaml(const char *filename)
    {

        YAML::Node config;
        try
        {
            config = YAML::LoadFile(filename);
        }
        catch (exception const &e)
        {
            LOG(ERROR) << "Error loading file, yaml file error or not exist." << filename;
            exit(-1);
        }

        try
        {
            YAML::Node frontend_conf = config["frontend_conf"];
            YAML::Node model_conf = config["model_conf"];
            this->sample_rate_ = frontend_conf["fs"].as<int>();  
            fbank_opts_.frame_opts.dither = 0;
            fbank_opts_.mel_opts.num_bins = model_conf["feat_dim"].as<int>();
            fbank_opts_.frame_opts.samp_freq = (float)this->sample_rate_ ;
            fbank_opts_.frame_opts.window_type = "povey";
            fbank_opts_.frame_opts.frame_shift_ms = 10;
            fbank_opts_.frame_opts.frame_length_ms = 25;
            fbank_opts_.energy_floor = 0;
            fbank_opts_.mel_opts.debug_mel = false;
        }
        catch (exception const &e)
        {
            LOG(ERROR) << "Error when load argument from campplus config YAML.";
            exit(-1);
        }
    }

    void CamPPlusSv::ReadModel(const char *cam_model)
    {
        try
        {
            cam_session_ = std::make_shared<Ort::Session>(
                env_, ORTCHAR(cam_model), session_options_);
            LOG(INFO) << "Successfully load model from " << cam_model;
        }
        catch (std::exception const &e)
        {
            LOG(ERROR) << "Error when load campplus onnx model: " << e.what();
            exit(-1);
        }
        GetInputOutputInfo(cam_session_, &cam_in_names_, &cam_out_names_);
    }

    void CamPPlusSv::GetInputOutputInfo(
        const std::shared_ptr<Ort::Session> &session,
        std::vector<const char *> *in_names, std::vector<const char *> *out_names)
    {
        Ort::AllocatorWithDefaultOptions allocator;
        // Input info
        int num_nodes = session->GetInputCount();
        in_names->resize(num_nodes);
        for (int i = 0; i < num_nodes; ++i)
        {
            std::unique_ptr<char, Ort::detail::AllocatedFree> name = session->GetInputNameAllocated(i, allocator);
            Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            ONNXTensorElementDataType type = tensor_info.GetElementType();
            std::vector<int64_t> node_dims = tensor_info.GetShape();
            std::stringstream shape;
            for (auto j : node_dims)
            {
                shape << j;
                shape << " ";
            }
            LOG(INFO) << "\tInput " << i << " : name=" << name.get() << " type=" << type
                      << " dims=" << shape.str();
            (*in_names)[i] = name.get();
            name.release();
        }
        // Output info
        num_nodes = session->GetOutputCount();
        out_names->resize(num_nodes);
        for (int i = 0; i < num_nodes; ++i)
        {
            std::unique_ptr<char, Ort::detail::AllocatedFree> name = session->GetOutputNameAllocated(i, allocator);
            Ort::TypeInfo type_info = session->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            ONNXTensorElementDataType type = tensor_info.GetElementType();
            std::vector<int64_t> node_dims = tensor_info.GetShape();
            std::stringstream shape;
            for (auto j : node_dims)
            {
                shape << j;
                shape << " ";
            }
            LOG(INFO) << "\tOutput " << i << " : name=" << name.get() << " type=" << type
                      << " dims=" << shape.str();
            (*out_names)[i] = name.get();
            name.release();
        }
    }

    void CamPPlusSv::Forward(
        const std::vector<std::vector<float>> &chunk_feats,
        std::vector<std::vector<float>> *out_prob)
    {

        Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        int num_frames = chunk_feats.size();           // 553
        const int feature_dim = chunk_feats[0].size(); // 80

        //  2. Generate input nodes tensor
        const int64_t vad_feats_shape[3] = {1, num_frames, feature_dim}; //[1,553,80]
        std::vector<float> vad_feats;
        for (const auto &chunk_feat : chunk_feats)
        {
            vad_feats.insert(vad_feats.end(), chunk_feat.begin(), chunk_feat.end());
        }
        Ort::Value vad_feats_ort = Ort::Value::CreateTensor<float>(
            memory_info, vad_feats.data(), vad_feats.size(), vad_feats_shape, 3);

        // 3. Put nodes into onnx input vector
        std::vector<Ort::Value> vad_inputs;
        vad_inputs.emplace_back(std::move(vad_feats_ort));

        // 4. Onnx infer
        std::vector<Ort::Value> vad_ort_outputs;
        try
        {
            vad_ort_outputs = cam_session_->Run(
                Ort::RunOptions{nullptr}, cam_in_names_.data(), vad_inputs.data(),
                vad_inputs.size(), cam_out_names_.data(), cam_out_names_.size());
        }
        catch (std::exception const &e)
        {
            LOG(ERROR) << "Error when run vad onnx forword: " << (e.what());
            return;
        }
        // 5. Change infer result to output shapes
        float *logp_data = vad_ort_outputs[0].GetTensorMutableData<float>();
        auto type_info = vad_ort_outputs[0].GetTensorTypeAndShapeInfo();

        int num_outputs = type_info.GetShape()[0]; //batch
        int output_dim = type_info.GetShape()[1]; //192
        out_prob->resize(num_outputs);
        for (int i = 0; i < num_outputs; i++)
        {
            (*out_prob)[i].resize(output_dim);
            memcpy((*out_prob)[i].data(), logp_data + i * output_dim,
                   sizeof(float) * output_dim);
        }
    }

    void CamPPlusSv::FbankKaldi(float sample_rate, std::vector<std::vector<float>> &vad_feats,
                                std::vector<float> &waves)
    {
        knf::OnlineFbank fbank(fbank_opts_);

        std::vector<float> buf(waves.size());
        for (int32_t i = 0; i != waves.size(); ++i)
        {
            buf[i] = waves[i] * 1;
        }
        fbank.AcceptWaveform(sample_rate, buf.data(), buf.size());
        int32_t frames = fbank.NumFramesReady();
        for (int32_t i = 0; i != frames; ++i)
        {
            const float *frame = fbank.GetFrame(i);
            std::vector<float> frame_vector(frame, frame + fbank_opts_.mel_opts.num_bins);
            vad_feats.emplace_back(frame_vector);
        }
    }

    void CamPPlusSv::LoadCmvn(const char *filename)
    {
        try
        {
            using namespace std;
            ifstream cmvn_stream(filename);
            if (!cmvn_stream.is_open())
            {
                LOG(ERROR) << "Failed to open file: " << filename;
                exit(-1);
            }
            string line;

            while (getline(cmvn_stream, line))
            {
                istringstream iss(line);
                vector<string> line_item{istream_iterator<string>{iss}, istream_iterator<string>{}};
                if (line_item[0] == "<AddShift>")
                {
                    getline(cmvn_stream, line);
                    istringstream means_lines_stream(line);
                    vector<string> means_lines{istream_iterator<string>{means_lines_stream}, istream_iterator<string>{}};
                    if (means_lines[0] == "<LearnRateCoef>")
                    {
                        for (int j = 3; j < means_lines.size() - 1; j++)
                        {
                            means_list_.push_back(stof(means_lines[j]));
                        }
                        continue;
                    }
                }
                else if (line_item[0] == "<Rescale>")
                {
                    getline(cmvn_stream, line);
                    istringstream vars_lines_stream(line);
                    vector<string> vars_lines{istream_iterator<string>{vars_lines_stream}, istream_iterator<string>{}};
                    if (vars_lines[0] == "<LearnRateCoef>")
                    {
                        for (int j = 3; j < vars_lines.size() - 1; j++)
                        {                           
                            vars_list_.push_back(stof(vars_lines[j]));
                        }
                        continue;
                    }
                }
            }
        }
        catch (std::exception const &e)
        {
            LOG(ERROR) << "Error when load vad cmvn : " << e.what();
            exit(-1);
        }
    }

    void CamPPlusSv::LfrCmvn(std::vector<std::vector<float>> &vad_feats)
    {

        // std::vector<std::vector<float>> out_feats;
        // int T = vad_feats.size();
        // int T_lrf = ceil(1.0 * T / lfr_n);

        // // Pad frames at start(copy first frame)
        // for (int i = 0; i < (lfr_m - 1) / 2; i++)
        // {
        //     vad_feats.insert(vad_feats.begin(), vad_feats[0]);
        // }
        // // Merge lfr_m frames as one,lfr_n frames per window
        // T = T + (lfr_m - 1) / 2;
        // std::vector<float> p;
        // for (int i = 0; i < T_lrf; i++)
        // {
        //     if (lfr_m <= T - i * lfr_n)
        //     {
        //         for (int j = 0; j < lfr_m; j++)
        //         {
        //             p.insert(p.end(), vad_feats[i * lfr_n + j].begin(), vad_feats[i * lfr_n + j].end());
        //         }
        //         out_feats.emplace_back(p);
        //         p.clear();
        //     }
        //     else
        //     {
        //         // Fill to lfr_m frames at last window if less than lfr_m frames  (copy last frame)
        //         int num_padding = lfr_m - (T - i * lfr_n);
        //         for (int j = 0; j < (vad_feats.size() - i * lfr_n); j++)
        //         {
        //             p.insert(p.end(), vad_feats[i * lfr_n + j].begin(), vad_feats[i * lfr_n + j].end());
        //         }
        //         for (int j = 0; j < num_padding; j++)
        //         {
        //             p.insert(p.end(), vad_feats[vad_feats.size() - 1].begin(), vad_feats[vad_feats.size() - 1].end());
        //         }
        //         out_feats.emplace_back(p);
        //         p.clear();
        //     }
        // }
        // //Apply cmvn
        std::vector<std::vector<float>> out_feats;
        out_feats = vad_feats;
        // only  Apply cmvn
        for (auto &out_feat : out_feats)
        {
            // for (int j = 0; j < means_list_.size(); j++) {
            for (int j = 0; j < std::min(out_feats[0].size(), means_list_.size()); j++)
            { 
                out_feat[j] = (out_feat[j] + means_list_[j]) * vars_list_[j];
            }
        }
        vad_feats = out_feats;
    }
    void  CamPPlusSv::SubMean(std::vector<std::vector<float>> &voice_feats)
    {
        if (voice_feats.size() > 0)
        {
            int rows = voice_feats.size();
            int clos = voice_feats[0].size();
            std::vector<float> feat_mean(clos, 0.0);
            for (int j = 0; j < clos; ++j)
            {
                float sum = 0.0;
                for (int i = 0; i < rows; ++i)
                {
                    sum += voice_feats[i][j];
                }
                feat_mean[j] = sum / (float)rows;
            }

            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < clos; ++j)
                {
                    voice_feats[i][j] -= feat_mean[j];
                }
            }
        }
    }

    std::vector<std::vector<float>> CamPPlusSv::Infer(std::vector<float> &waves)
    {
        std::vector<std::vector<float>> vad_feats;
        std::vector<std::vector<float>> voice_features;

        FbankKaldi(sample_rate_, vad_feats, waves);

        if (vad_feats.size() == 0)
        {
            return voice_features;
        }
        // sub mean  pad 
        SubMean(vad_feats);  
        Forward(vad_feats, &voice_features);  
        return voice_features;
    }

    CamPPlusSv::~CamPPlusSv()
    {
    }

    CamPPlusSv::CamPPlusSv() : env_(ORT_LOGGING_LEVEL_ERROR, ""), session_options_{}
    {
    }

} // namespace funasrr<std: <:vector<float   st d::vector<std::vector<float>> vad_probs;

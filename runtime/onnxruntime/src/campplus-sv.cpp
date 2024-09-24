/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
 */

#include <fstream>
#include "precomp.h"
#include <vector>

namespace funasr
{
    void CamPPlusSv::InitSv(const std::string &model, const std::string &config, int thread_num)
    {
        session_options_.SetIntraOpNumThreads(thread_num);
        session_options_.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
        session_options_.DisableCpuMemArena();

        ReadModel(model.c_str());
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
            // LOG(INFO) << "\tInput " << i << " : name=" << name.get() << " type=" << type
            //           << " dims=" << shape.str();
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
            // LOG(INFO) << "\tOutput " << i << " : name=" << name.get() << " type=" << type
            //           << " dims=" << shape.str();
            (*out_names)[i] = name.get();
            name.release();
        }
    }

    void CamPPlusSv::Forward(
        const std::vector<std::vector<std::vector<float>>> &chunk_feats,
        std::vector<std::vector<float>> *out_prob)
    {

        Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        int batch_size = chunk_feats.size();
        int num_frames = chunk_feats[0].size();           // 553
        const int feature_dim = chunk_feats[0][0].size(); // 80

        //  2. Generate input nodes tensor
        const int64_t vad_feats_shape[3] = {batch_size, num_frames, feature_dim}; //[1,553,80]
        std::vector<float> vad_feats;
        for (const auto &chunk_feat_b : chunk_feats)
        {
            for (const auto &chunk_feat : chunk_feat_b)
            {
                vad_feats.insert(vad_feats.end(), chunk_feat.begin(), chunk_feat.end());
            }
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

    std::vector<std::vector<float>> CamPPlusSv::Infer(sv_segment vad_seg)
    {
        std::vector<std::vector<float>> voice_features;

        std::vector<std::vector<std::vector<float>>> vad_feats;
        std::vector<sv_segment> seg_res = SegChunk(vad_seg);
        for(int idx=0; idx<seg_res.size(); idx++){
            std::vector<std::vector<float>> vad_feat;
            std::vector<float> wave = seg_res[idx].data;
            FbankKaldi(sample_rate_, vad_feat, wave);
            SubMean(vad_feat);
            vad_feats.push_back(vad_feat);
        }

        if (vad_feats.size() == 0 || 
            vad_feats[0].size() == 0 ||
            vad_feats[0][0].size() == 0){
            return voice_features;
        }
        Forward(vad_feats, &voice_features);
        return voice_features;
    }

    std::vector<sv_segment> CamPPlusSv::SegChunk(sv_segment& seg_data) {
        double seg_st = seg_data.start_time;
        const std::vector<float>& data = seg_data.data;

        int chunk_len = static_cast<int>(seg_dur * (double)sample_rate_);
        int chunk_shift = static_cast<int>(seg_shift * (double)sample_rate_);
        int last_chunk_ed = 0;

        std::vector<sv_segment> seg_res;

        for (int chunk_st = 0; chunk_st < data.size(); chunk_st += chunk_shift) {
            int chunk_ed = std::min(chunk_st + chunk_len, static_cast<int>(data.size()));
            if (chunk_ed <= last_chunk_ed) {
                break;
            }
            last_chunk_ed = chunk_ed;
            chunk_st = std::max(0, chunk_ed - chunk_len);

            std::vector<float> chunk_data(data.begin() + chunk_st, data.begin() + chunk_ed);

            if (chunk_data.size() < chunk_len) {
                chunk_data.resize(chunk_len, 0.0f);
            }

            seg_res.push_back({
                (double)chunk_st / (double)sample_rate_ + seg_st,
                (double)chunk_ed / (double)sample_rate_ + seg_st,
                chunk_data
            });
        }
        return seg_res;
    }

    CamPPlusSv::~CamPPlusSv()
    {
    }

    CamPPlusSv::CamPPlusSv() : env_(ORT_LOGGING_LEVEL_ERROR, ""), session_options_{}
    {
    }

} // namespace funasrr<std: <:vector<float   st d::vector<std::vector<float>> vad_probs;

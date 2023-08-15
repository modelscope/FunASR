/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/

#include "precomp.h"
#include "paraformer.h"
#include "encode_converter.h"
#include <cstddef>

using namespace std;
static int embedding_dim = 512;
namespace funasr {

Paraformer::Paraformer()
:use_hotword(false),
 env_(ORT_LOGGING_LEVEL_ERROR, "paraformer"),session_options{},
 hw_env_(ORT_LOGGING_LEVEL_ERROR, "paraformer_hw"),hw_session_options{} {
}


void Paraformer::InitAsr(const std::string &am_model, const std::string &am_cmvn, const std::string &am_config, int thread_num){
    // knf options
    fbank_opts.frame_opts.dither = 0;
    fbank_opts.mel_opts.num_bins = 80;
    fbank_opts.frame_opts.samp_freq = MODEL_SAMPLE_RATE;
    fbank_opts.frame_opts.window_type = "hamming";
    fbank_opts.frame_opts.frame_shift_ms = 10;
    fbank_opts.frame_opts.frame_length_ms = 25;
    fbank_opts.energy_floor = 0;
    fbank_opts.mel_opts.debug_mel = false;
    // fbank_ = std::make_unique<knf::OnlineFbank>(fbank_opts);

    // session_options.SetInterOpNumThreads(1);
    session_options.SetIntraOpNumThreads(thread_num);
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    // DisableCpuMemArena can improve performance
    session_options.DisableCpuMemArena();

    try {
        m_session = std::make_unique<Ort::Session>(env_, am_model.c_str(), session_options);
        LOG(INFO) << "Successfully load model from " << am_model;
    } catch (std::exception const &e) {
        LOG(ERROR) << "Error when load am onnx model: " << e.what();
        exit(0);
    }

    string strName;
    GetInputName(m_session.get(), strName);
    m_strInputNames.push_back(strName.c_str());
    GetInputName(m_session.get(), strName,1);
    m_strInputNames.push_back(strName);
    if (use_hotword) {
        GetInputName(m_session.get(), strName, 2);
        m_strInputNames.push_back(strName);
    }
    
    GetOutputName(m_session.get(), strName);
    m_strOutputNames.push_back(strName);
    GetOutputName(m_session.get(), strName,1);
    m_strOutputNames.push_back(strName);

    for (auto& item : m_strInputNames)
        m_szInputNames.push_back(item.c_str());
    for (auto& item : m_strOutputNames)
        m_szOutputNames.push_back(item.c_str());
    vocab = new Vocab(am_config.c_str());
    LoadCmvn(am_cmvn.c_str());
}

void Paraformer::InitHwCompiler(const std::string &hw_model, int thread_num) {
    hw_session_options.SetIntraOpNumThreads(thread_num);
    hw_session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    // DisableCpuMemArena can improve performance
    hw_session_options.DisableCpuMemArena();

    try {
        hw_m_session = std::make_unique<Ort::Session>(hw_env_, hw_model.c_str(), hw_session_options);
        LOG(INFO) << "Successfully load model from " << hw_model;
    } catch (std::exception const &e) {
        LOG(ERROR) << "Error when load hw compiler onnx model: " << e.what();
        exit(0);
    }

    string strName;
    GetInputName(hw_m_session.get(), strName);
    hw_m_strInputNames.push_back(strName.c_str());
    //GetInputName(hw_m_session.get(), strName,1);
    //hw_m_strInputNames.push_back(strName);
    
    GetOutputName(hw_m_session.get(), strName);
    hw_m_strOutputNames.push_back(strName);

    for (auto& item : hw_m_strInputNames)
        hw_m_szInputNames.push_back(item.c_str());
    for (auto& item : hw_m_strOutputNames)
        hw_m_szOutputNames.push_back(item.c_str());
    // if init hotword compiler is called, this is a hotword paraformer model
    use_hotword = true;
}

void Paraformer::InitSegDict(const std::string &seg_dict_model) {
    seg_dict = new SegDict(seg_dict_model.c_str());
}

Paraformer::~Paraformer()
{
    if(vocab)
        delete vocab;
    if(seg_dict)
        delete seg_dict;
}

void Paraformer::Reset()
{
}

vector<float> Paraformer::FbankKaldi(float sample_rate, const float* waves, int len) {
    knf::OnlineFbank fbank_(fbank_opts);
    std::vector<float> buf(len);
    for (int32_t i = 0; i != len; ++i) {
        buf[i] = waves[i] * 32768;
    }
    fbank_.AcceptWaveform(sample_rate, buf.data(), buf.size());
    //fbank_->InputFinished();
    int32_t frames = fbank_.NumFramesReady();
    int32_t feature_dim = fbank_opts.mel_opts.num_bins;
    vector<float> features(frames * feature_dim);
    float *p = features.data();
    //std::cout << "samples " << len << std::endl;
    //std::cout << "fbank frames " << frames << std::endl;
    //std::cout << "fbank dim " << feature_dim << std::endl;
    //std::cout << "feature size " << features.size() << std::endl;

    for (int32_t i = 0; i != frames; ++i) {
        const float *f = fbank_.GetFrame(i);
        std::copy(f, f + feature_dim, p);
        p += feature_dim;
    }

    return features;
}

void Paraformer::LoadCmvn(const char *filename)
{
    ifstream cmvn_stream(filename);
    if (!cmvn_stream.is_open()) {
        LOG(ERROR) << "Failed to open file: " << filename;
        exit(0);
    }
    string line;

    while (getline(cmvn_stream, line)) {
        istringstream iss(line);
        vector<string> line_item{istream_iterator<string>{iss}, istream_iterator<string>{}};
        if (line_item[0] == "<AddShift>") {
            getline(cmvn_stream, line);
            istringstream means_lines_stream(line);
            vector<string> means_lines{istream_iterator<string>{means_lines_stream}, istream_iterator<string>{}};
            if (means_lines[0] == "<LearnRateCoef>") {
                for (int j = 3; j < means_lines.size() - 1; j++) {
                    means_list.push_back(stof(means_lines[j]));
                }
                continue;
            }
        }
        else if (line_item[0] == "<Rescale>") {
            getline(cmvn_stream, line);
            istringstream vars_lines_stream(line);
            vector<string> vars_lines{istream_iterator<string>{vars_lines_stream}, istream_iterator<string>{}};
            if (vars_lines[0] == "<LearnRateCoef>") {
                for (int j = 3; j < vars_lines.size() - 1; j++) {
                    vars_list.push_back(stof(vars_lines[j])*scale);
                }
                continue;
            }
        }
    }
}

string Paraformer::GreedySearch(float * in, int n_len,  int64_t token_nums)
{
    vector<int> hyps;
    int Tmax = n_len;
    for (int i = 0; i < Tmax; i++) {
        int max_idx;
        float max_val;
        FindMax(in + i * token_nums, token_nums, max_val, max_idx);
        hyps.push_back(max_idx);
    }

    return vocab->Vector2StringV2(hyps);
}

vector<float> Paraformer::ApplyLfr(const std::vector<float> &in) 
{
    int32_t in_feat_dim = fbank_opts.mel_opts.num_bins;
    int32_t in_num_frames = in.size() / in_feat_dim;
    int32_t out_num_frames =
        (in_num_frames - lfr_window_size) / lfr_window_shift + 1;
    int32_t out_feat_dim = in_feat_dim * lfr_window_size;

    std::vector<float> out(out_num_frames * out_feat_dim);

    const float *p_in = in.data();
    float *p_out = out.data();

    for (int32_t i = 0; i != out_num_frames; ++i) {
      std::copy(p_in, p_in + out_feat_dim, p_out);

      p_out += out_feat_dim;
      p_in += lfr_window_shift * in_feat_dim;
    }

    return out;
  }

  void Paraformer::ApplyCmvn(std::vector<float> *v)
  {
    int32_t dim = means_list.size();
    int32_t num_frames = v->size() / dim;

    float *p = v->data();

    for (int32_t i = 0; i != num_frames; ++i) {
      for (int32_t k = 0; k != dim; ++k) {
        p[k] = (p[k] + means_list[k]) * vars_list[k];
      }

      p += dim;
    }
  }

string Paraformer::Forward(float* din, int len, int flag, const std::vector<std::vector<float>> &hw_emb)
{

    int32_t in_feat_dim = fbank_opts.mel_opts.num_bins;
    //std::cout << "pcm len " << len << std::endl;
    std::vector<float> wav_feats = FbankKaldi(MODEL_SAMPLE_RATE, din, len);
    wav_feats = ApplyLfr(wav_feats);
    ApplyCmvn(&wav_feats);

    int32_t feat_dim = lfr_window_size*in_feat_dim;
    int32_t num_frames = wav_feats.size() / feat_dim;
    //std::cout << "feat in: " << num_frames << " " << feat_dim << std::endl;

#ifdef _WIN_X86
        Ort::MemoryInfo m_memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
#else
        Ort::MemoryInfo m_memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
#endif

    const int64_t input_shape_[3] = {1, num_frames, feat_dim};
    Ort::Value onnx_feats = Ort::Value::CreateTensor<float>(m_memoryInfo,
        wav_feats.data(),
        wav_feats.size(),
        input_shape_,
        3);

    const int64_t paraformer_length_shape[1] = {1};
    std::vector<int32_t> paraformer_length;
    paraformer_length.emplace_back(num_frames);
    Ort::Value onnx_feats_len = Ort::Value::CreateTensor<int32_t>(
          m_memoryInfo, paraformer_length.data(), paraformer_length.size(), paraformer_length_shape, 1);

    std::vector<Ort::Value> input_onnx;
    input_onnx.emplace_back(std::move(onnx_feats));
    input_onnx.emplace_back(std::move(onnx_feats_len));

    std::vector<float> embedding;
    if (use_hotword) {
      //PrintMat(hw_emb, "input_clas_emb");
      const int64_t hotword_shape[3] = {1, hw_emb.size(), hw_emb[0].size()};
      embedding.reserve(hw_emb.size() * hw_emb[0].size());
      for (auto item : hw_emb) {
        embedding.insert(embedding.end(), item.begin(), item.end());
      }
      //LOG(INFO) << "hotword shape " << hotword_shape[0] << " " << hotword_shape[1] << " " << hotword_shape[2] << " size " << embedding.size();
      Ort::Value onnx_hw_emb = Ort::Value::CreateTensor<float>(
          m_memoryInfo, embedding.data(), embedding.size(), hotword_shape, 3);

      input_onnx.emplace_back(std::move(onnx_hw_emb));
    }

    string result;
    try {
        auto outputTensor = m_session->Run(Ort::RunOptions{nullptr}, m_szInputNames.data(), input_onnx.data(), input_onnx.size(), m_szOutputNames.data(), m_szOutputNames.size());
        std::vector<int64_t> outputShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();
        //LOG(INFO) << "paraformer out shape " << outputShape[0] << " " << outputShape[1] << " " << outputShape[2];

        int64_t outputCount = std::accumulate(outputShape.begin(), outputShape.end(), 1, std::multiplies<int64_t>());
        float* floatData = outputTensor[0].GetTensorMutableData<float>();
        auto encoder_out_lens = outputTensor[1].GetTensorMutableData<int64_t>();
        int pos = 0;
        std::vector<std::vector<float>> logits;
        for (int j = 0; j < outputShape[1]; j++)
        {
            std::vector<float> vec_token;
            vec_token.insert(vec_token.begin(), floatData + pos, floatData + pos + outputShape[2]);
            logits.push_back(vec_token);
            pos += outputShape[2];
        }
        //PrintMat(logits, "logits_out");
        result = GreedySearch(floatData, *encoder_out_lens, outputShape[2]);
    }
    catch (std::exception const &e)
    {
        LOG(ERROR)<<e.what();
    }

    return result;
}


std::vector<std::vector<float>> Paraformer::CompileHotwordEmbedding(std::string &hotwords) {
    std::vector<std::vector<float>> hw_emb;
    if (!use_hotword) {
        std::vector<float> vec(embedding_dim, 0);
        hw_emb.push_back(vec);
        return hw_emb;
    }
    int max_hotword_len = 10;
    std::vector<int32_t> hotword_matrix;
    std::vector<int32_t> lengths;
    int hotword_size = 1;
    if (!hotwords.empty()) {
      std::vector<std::string> hotword_array = split(hotwords, '|');
      hotword_size = hotword_array.size() + 1;
      hotword_matrix.reserve(hotword_size * max_hotword_len);
      for (auto hotword : hotword_array) {
        std::vector<std::string> chars;
        if (EncodeConverter::IsAllChineseCharactor((const U8CHAR_T*)hotword.c_str(), hotword.size())) {
          KeepChineseCharacterAndSplit(hotword, chars);
        } else {
          // for english
          std::vector<std::string> words = split(hotword, ' ');
          for (auto word : words) {
            std::vector<string> tokens = seg_dict->GetTokensByWord(word);
            chars.insert(chars.end(), tokens.begin(), tokens.end());
          }
        }
        std::vector<int32_t> hw_vector(max_hotword_len, 0);
        int vector_len = std::min(max_hotword_len, (int)chars.size());
        for (int i=0; i<chars.size(); i++) {
          std::cout << chars[i] << " ";
          hw_vector[i] = vocab->GetIdByToken(chars[i]);
        }
        std::cout << std::endl;
        lengths.push_back(vector_len);
        hotword_matrix.insert(hotword_matrix.end(), hw_vector.begin(), hw_vector.end());
      }
    }
    std::vector<int32_t> blank_vec(max_hotword_len, 0);
    blank_vec[0] = 1;
    hotword_matrix.insert(hotword_matrix.end(), blank_vec.begin(), blank_vec.end());
    lengths.push_back(1);

#ifdef _WIN_X86
        Ort::MemoryInfo m_memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
#else
        Ort::MemoryInfo m_memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
#endif

    const int64_t input_shape_[2] = {hotword_size, max_hotword_len};
    Ort::Value onnx_hotword = Ort::Value::CreateTensor<int32_t>(m_memoryInfo,
        (int32_t*)hotword_matrix.data(),
        hotword_size * max_hotword_len,
        input_shape_,
        2);
    LOG(INFO) << "clas shape " << hotword_size << " " << max_hotword_len << std::endl;
    
    std::vector<Ort::Value> input_onnx;
    input_onnx.emplace_back(std::move(onnx_hotword));

    std::vector<std::vector<float>> result;
    try {
        auto outputTensor = hw_m_session->Run(Ort::RunOptions{nullptr}, hw_m_szInputNames.data(), input_onnx.data(), input_onnx.size(), hw_m_szOutputNames.data(), hw_m_szOutputNames.size());
        std::vector<int64_t> outputShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();

        int64_t outputCount = std::accumulate(outputShape.begin(), outputShape.end(), 1, std::multiplies<int64_t>());
        float* floatData = outputTensor[0].GetTensorMutableData<float>(); // shape [max_hotword_len, hotword_size, dim]
        // get embedding by real hotword length
        assert(outputShape[0] == max_hotword_len);
        assert(outputShape[1] == hotword_size);
        embedding_dim = outputShape[2];

        for (int j = 0; j < hotword_size; j++)
        {
            int start_pos = hotword_size * (lengths[j] - 1) * embedding_dim + j * embedding_dim;
            std::vector<float> embedding;
            embedding.insert(embedding.begin(), floatData + start_pos, floatData + start_pos + embedding_dim);
            result.push_back(embedding);
        }
    }
    catch (std::exception const &e)
    {
        LOG(ERROR)<<e.what();
    }
    //PrintMat(result, "clas_embedding_output");
    return result;
}

string Paraformer::ForwardChunk(float* din, int len, int flag)
{

    LOG(ERROR)<<"Not Imp!!!!!!";
    return "";
}

string Paraformer::Rescoring()
{
    LOG(ERROR)<<"Not Imp!!!!!!";
    return "";
}
} // namespace funasr

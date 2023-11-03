
#include "precomp.h"

namespace funasr {
float *LoadParams(const char *filename)
{

    FILE *fp;
    fp = fopen(filename, "rb");
    fseek(fp, 0, SEEK_END);
    uint32_t nFileLen = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    float *params_addr = (float *)AlignedMalloc(32, nFileLen);
    int n = fread(params_addr, 1, nFileLen, fp);
    fclose(fp);

    return params_addr;
}

int ValAlign(int val, int align)
{
    float tmp = ceil((float)val / (float)align) * (float)align;
    return (int)tmp;
}

void DispParams(float *din, int size)
{
    int i;
    for (i = 0; i < size; i++) {
        printf("%f ", din[i]);
    }
    printf("\n");
}
void SaveDataFile(const char *filename, void *data, uint32_t len)
{
    FILE *fp;
    fp = fopen(filename, "wb+");
    fwrite(data, 1, len, fp);
    fclose(fp);
}

void BasicNorm(Tensor<float> *&din, float norm)
{

    int Tmax = din->size[2];

    int i, j;
    for (i = 0; i < Tmax; i++) {
        float sum = 0;
        for (j = 0; j < 512; j++) {
            int ii = i * 512 + j;
            sum += din->buff[ii] * din->buff[ii];
        }
        float mean = sqrt(sum / 512 + norm);
        for (j = 0; j < 512; j++) {
            int ii = i * 512 + j;
            din->buff[ii] = din->buff[ii] / mean;
        }
    }
}

void FindMax(float *din, int len, float &max_val, int &max_idx)
{
    int i;
    max_val = -INFINITY;
    max_idx = -1;
    for (i = 0; i < len; i++) {
        if (din[i] > max_val) {
            max_val = din[i];
            max_idx = i;
        }
    }
}

string PathAppend(const string &p1, const string &p2)
{

    char sep = '/';
    string tmp = p1;

#ifdef _WIN32
    sep = '\\';
#endif

    if (p1[p1.length()-1] != sep) { // Need to add a
        tmp += sep;               // path separator
        return (tmp + p2);
    } else
        return (p1 + p2);
}

void Relu(Tensor<float> *din)
{
    int i;
    for (i = 0; i < din->buff_size; i++) {
        float val = din->buff[i];
        din->buff[i] = val < 0 ? 0 : val;
    }
}

void Swish(Tensor<float> *din)
{
    int i;
    for (i = 0; i < din->buff_size; i++) {
        float val = din->buff[i];
        din->buff[i] = val / (1 + exp(-val));
    }
}

void Sigmoid(Tensor<float> *din)
{
    int i;
    for (i = 0; i < din->buff_size; i++) {
        float val = din->buff[i];
        din->buff[i] = 1 / (1 + exp(-val));
    }
}

void DoubleSwish(Tensor<float> *din)
{
    int i;
    for (i = 0; i < din->buff_size; i++) {
        float val = din->buff[i];
        din->buff[i] = val / (1 + exp(-val + 1));
    }
}

void Softmax(float *din, int mask, int len)
{
    float *tmp = (float *)malloc(mask * sizeof(float));
    int i;
    float sum = 0;
    float max = -INFINITY;

    for (i = 0; i < mask; i++) {
        max = max < din[i] ? din[i] : max;
    }

    for (i = 0; i < mask; i++) {
        tmp[i] = exp(din[i] - max);
        sum += tmp[i];
    }
    for (i = 0; i < mask; i++) {
        din[i] = tmp[i] / sum;
    }
    free(tmp);
    for (i = mask; i < len; i++) {
        din[i] = 0;
    }
}

void LogSoftmax(float *din, int len)
{
    float *tmp = (float *)malloc(len * sizeof(float));
    int i;
    float sum = 0;
    for (i = 0; i < len; i++) {
        tmp[i] = exp(din[i]);
        sum += tmp[i];
    }
    for (i = 0; i < len; i++) {
        din[i] = log(tmp[i] / sum);
    }
    free(tmp);
}

void Glu(Tensor<float> *din, Tensor<float> *dout)
{
    int mm = din->buff_size / 1024;
    int i, j;
    for (i = 0; i < mm; i++) {
        for (j = 0; j < 512; j++) {
            int in_off = i * 1024 + j;
            int out_off = i * 512 + j;
            float a = din->buff[in_off];
            float b = din->buff[in_off + 512];
            dout->buff[out_off] = a / (1 + exp(-b));
        }
    }
}

bool is_target_file(const std::string& filename, const std::string target) {
    std::size_t pos = filename.find_last_of(".");
    if (pos == std::string::npos) {
        return false;
    }
    std::string extension = filename.substr(pos + 1);
    return (extension == target);
}

void KeepChineseCharacterAndSplit(const std::string &input_str,
                                  std::vector<std::string> &chinese_characters) {
  chinese_characters.resize(0);
  std::vector<U16CHAR_T> u16_buf;
  u16_buf.resize(std::max(u16_buf.size(), input_str.size() + 1));
  U16CHAR_T* pu16 = u16_buf.data();
  U8CHAR_T * pu8 = (U8CHAR_T*)input_str.data();
  size_t ilen = input_str.size();
  size_t len = EncodeConverter::Utf8ToUtf16(pu8, ilen, pu16, ilen + 1);
  for (size_t i = 0; i < len; i++) {
    if (EncodeConverter::IsChineseCharacter(pu16[i])) {
      U8CHAR_T u8buf[4];
      size_t n = EncodeConverter::Utf16ToUtf8(pu16 + i, u8buf);
      u8buf[n] = '\0';
      chinese_characters.push_back((const char*)u8buf);
    }
  }
}

void SplitChiEngCharacters(const std::string &input_str,
                                  std::vector<std::string> &characters) {
  characters.resize(0);
  std::string eng_word = "";
  U16CHAR_T space = 0x0020;
  std::vector<U16CHAR_T> u16_buf;
  u16_buf.resize(std::max(u16_buf.size(), input_str.size() + 1));
  U16CHAR_T* pu16 = u16_buf.data();
  U8CHAR_T * pu8 = (U8CHAR_T*)input_str.data();
  size_t ilen = input_str.size();
  size_t len = EncodeConverter::Utf8ToUtf16(pu8, ilen, pu16, ilen + 1);
  for (size_t i = 0; i < len; i++) {
    if (EncodeConverter::IsChineseCharacter(pu16[i])) {
      if(!eng_word.empty()){
        characters.push_back(eng_word);
        eng_word = "";
      }
      U8CHAR_T u8buf[4];
      size_t n = EncodeConverter::Utf16ToUtf8(pu16 + i, u8buf);
      u8buf[n] = '\0';
      characters.push_back((const char*)u8buf);
    } else if (pu16[i] == space){
      if(!eng_word.empty()){
        characters.push_back(eng_word);
        eng_word = "";
      }      
    }else{
      U8CHAR_T u8buf[4];
      size_t n = EncodeConverter::Utf16ToUtf8(pu16 + i, u8buf);
      u8buf[n] = '\0';
      eng_word += (const char*)u8buf;
    }
  }
  if(!eng_word.empty()){
    characters.push_back(eng_word);
    eng_word = "";
  }
}

std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> elems;
  std::stringstream ss(s);
  std::string item;
  while(std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
  return elems;
}

template<typename T>
void PrintMat(const std::vector<std::vector<T>> &mat, const std::string &name) {
  std::cout << name << ":" << std::endl;
  for (auto item : mat) {
    for (auto item_ : item) {
      std::cout << item_ << " ";
    }
    std::cout << std::endl;
  }
}

size_t Utf8ToCharset(const std::string &input, std::vector<std::string> &output) {
  std::string ch; 
  for (size_t i = 0, len = 0; i != input.length(); i += len) {
    unsigned char byte = (unsigned)input[i];
    if (byte >= 0xFC) // lenght 6
      len = 6;  
    else if (byte >= 0xF8)
      len = 5;
    else if (byte >= 0xF0)
      len = 4;
    else if (byte >= 0xE0)
      len = 3;
    else if (byte >= 0xC0)
      len = 2;
    else
      len = 1;
    ch = input.substr(i, len);
    output.push_back(ch);
  }   
  return output.size();
}

int Str2IntFunc(string str)
{
    const char *ch_array = str.c_str();
    if (((ch_array[0] & 0xf0) != 0xe0) || ((ch_array[1] & 0xc0) != 0x80) ||
        ((ch_array[2] & 0xc0) != 0x80))
        return 0;
    int val = ((ch_array[0] & 0x0f) << 12) | ((ch_array[1] & 0x3f) << 6) |
              (ch_array[2] & 0x3f);
    return val;
}

bool IsChinese(string ch)
{
    if (ch.size() != 3) {
        return false;
    }
    int unicode = Str2IntFunc(ch);
    if (unicode >= 19968 && unicode <= 40959) {
        return true;
    }
    return false;
}

string PostProcess(std::vector<string> &raw_char, std::vector<std::vector<float>> &timestamp_list){
    std::vector<std::vector<float>> timestamp_merge;
    int i;
    list<string> words;
    int is_pre_english = false;
    int pre_english_len = 0;
    int is_combining = false;
    string combine = "";

    float begin=-1;
    for (i=0; i<raw_char.size(); i++){
        string word = raw_char[i];
        // step1 space character skips
        if (word == "<s>" || word == "</s>" || word == "<unk>")
            continue;
        // step2 combie phoneme to full word
        {
            int sub_word = !(word.find("@@") == string::npos);
            // process word start and middle part
            if (sub_word) {
                combine += word.erase(word.length() - 2);
                if(!is_combining){
                    begin = timestamp_list[i][0];
                }
                is_combining = true;
                continue;
            }
            // process word end part
            else if (is_combining) {
                combine += word;
                is_combining = false;
                word = combine;
                combine = "";
            }
        }

        // step3 process english word deal with space , turn abbreviation to upper case
        {
            // input word is chinese, not need process 
            if (IsChinese(word)) {
                words.push_back(word);
                timestamp_merge.emplace_back(timestamp_list[i]);
                is_pre_english = false;
            }
            // input word is english word
            else {
                // pre word is chinese
                if (!is_pre_english) {
                    // word[0] = word[0] - 32;
                    words.push_back(word);
                    begin = (begin==-1)?timestamp_list[i][0]:begin;
                    std::vector<float> vec = {begin, timestamp_list[i][1]};
                    timestamp_merge.emplace_back(vec);
                    begin = -1;
                    pre_english_len = word.size();
                }
                // pre word is english word
                else {
                    // single letter turn to upper case
                    // if (word.size() == 1) {
                    //     word[0] = word[0] - 32;
                    // }

                    if (pre_english_len > 1) {
                        words.push_back(" ");
                        words.push_back(word);
                        begin = (begin==-1)?timestamp_list[i][0]:begin;
                        std::vector<float> vec = {begin, timestamp_list[i][1]};
                        timestamp_merge.emplace_back(vec);
                        begin = -1;
                        pre_english_len = word.size();
                    }
                    else {
                        // if (word.size() > 1) {
                        //     words.push_back(" ");
                        // }
                        words.push_back(" ");
                        words.push_back(word);
                        begin = (begin==-1)?timestamp_list[i][0]:begin;
                        std::vector<float> vec = {begin, timestamp_list[i][1]};
                        timestamp_merge.emplace_back(vec);
                        begin = -1;
                        pre_english_len = word.size();
                    }
                }
                is_pre_english = true;
            }
        }
    }
    string stamp_str="";
    for (i=0; i<timestamp_merge.size(); i++) {
        stamp_str += std::to_string(timestamp_merge[i][0]);
        stamp_str += ", ";
        stamp_str += std::to_string(timestamp_merge[i][1]);
        if(i!=timestamp_merge.size()-1){
            stamp_str += ",";
        }
    }

    stringstream ss;
    for (auto it = words.begin(); it != words.end(); it++) {
        ss << *it;
    }

    return ss.str()+" | "+stamp_str;
}

void TimestampOnnx( std::vector<float>& us_alphas,
                    std::vector<float> us_cif_peak, 
                    std::vector<string>& char_list, 
                    std::string &res_str, 
                    std::vector<std::vector<float>> &timestamp_vec, 
                    float begin_time, 
                    float total_offset){
    if (char_list.empty()) {
        return ;
    }

    const float START_END_THRESHOLD = 5.0;
    const float MAX_TOKEN_DURATION = 30.0;
    const float TIME_RATE = 10.0 * 6 / 1000 / 3;
    // 3 times upsampled, cif_peak is flattened into a 1D array
    std::vector<float> cif_peak = us_cif_peak;
    int num_frames = cif_peak.size();
    if (char_list.back() == "</s>") {
        char_list.pop_back();
    }
    if (char_list.empty()) {
        return ;
    }
    vector<vector<float>> timestamp_list;
    vector<string> new_char_list;
    vector<float> fire_place;
    // for bicif model trained with large data, cif2 actually fires when a character starts
    // so treat the frames between two peaks as the duration of the former token
    for (int i = 0; i < num_frames; i++) {
        if (cif_peak[i] > 1.0 - 1e-4) {
            fire_place.push_back(i + total_offset);
        }
    }
    int num_peak = fire_place.size();
    if(num_peak != (int)char_list.size() + 1){
        float sum = std::accumulate(us_alphas.begin(), us_alphas.end(), 0.0f);
        float scale = sum/((int)char_list.size() + 1);
        if(scale == 0){
            return;
        }
        cif_peak.clear();
        sum = 0.0;
        for(auto &alpha:us_alphas){
            alpha = alpha/scale;
            sum += alpha;
            cif_peak.emplace_back(sum);
            if(sum>=1.0 - 1e-4){
                sum -=(1.0 - 1e-4);
            }            
        }

        fire_place.clear();
        for (int i = 0; i < num_frames; i++) {
            if (cif_peak[i] > 1.0 - 1e-4) {
                fire_place.push_back(i + total_offset);
            }
        }
    }
    
    num_peak = fire_place.size();
    if(fire_place.size() == 0){
        return;
    }

    // begin silence
    if (fire_place[0] > START_END_THRESHOLD) {
        new_char_list.push_back("<sil>");
        timestamp_list.push_back({0.0, fire_place[0] * TIME_RATE});
    }

    // tokens timestamp
    for (int i = 0; i < num_peak - 1; i++) {
        new_char_list.push_back(char_list[i]);
        if (i == num_peak - 2 || MAX_TOKEN_DURATION < 0 || fire_place[i + 1] - fire_place[i] < MAX_TOKEN_DURATION) {
            timestamp_list.push_back({fire_place[i] * TIME_RATE, fire_place[i + 1] * TIME_RATE});
        } else {
            // cut the duration to token and sil of the 0-weight frames last long
            float _split = fire_place[i] + MAX_TOKEN_DURATION;
            timestamp_list.push_back({fire_place[i] * TIME_RATE, _split * TIME_RATE});
            timestamp_list.push_back({_split * TIME_RATE, fire_place[i + 1] * TIME_RATE});
            new_char_list.push_back("<sil>");
        }
    }

    // tail token and end silence
    if(timestamp_list.size()==0){
        LOG(ERROR)<<"timestamp_list's size is 0!";
        return;
    }
    if (num_frames - fire_place.back() > START_END_THRESHOLD) {
        float _end = (num_frames + fire_place.back()) / 2.0;
        timestamp_list.back()[1] = _end * TIME_RATE;
        timestamp_list.push_back({_end * TIME_RATE, num_frames * TIME_RATE});
        new_char_list.push_back("<sil>");
    } else {
        timestamp_list.back()[1] = num_frames * TIME_RATE;
    }

    if (begin_time) {  // add offset time in model with vad
        for (auto& timestamp : timestamp_list) {
            timestamp[0] += begin_time / 1000.0;
            timestamp[1] += begin_time / 1000.0;
        }
    }

    assert(new_char_list.size() == timestamp_list.size());

    for (int i = 0; i < (int)new_char_list.size(); i++) {
        res_str += new_char_list[i] + " " + to_string(timestamp_list[i][0]) + " " + to_string(timestamp_list[i][1]) + ";";
    }

    for (int i = 0; i < (int)new_char_list.size(); i++) {
        if(new_char_list[i] != "<sil>"){
            timestamp_vec.push_back(timestamp_list[i]);
        }
    }
}

bool IsTargetFile(const std::string& filename, const std::string target) {
    std::size_t pos = filename.find_last_of(".");
    if (pos == std::string::npos) {
        return false;
    }
    std::string extension = filename.substr(pos + 1);
    return (extension == target);
}

void Trim(std::string *str) {
  const char *white_chars = " \t\n\r\f\v";

  std::string::size_type pos = str->find_last_not_of(white_chars);
  if (pos != std::string::npos)  {
    str->erase(pos + 1);
    pos = str->find_first_not_of(white_chars);
    if (pos != std::string::npos) str->erase(0, pos);
  } else {
    str->erase(str->begin(), str->end());
  }
}

void SplitStringToVector(const std::string &full, const char *delim,
                         bool omit_empty_strings,
                         std::vector<std::string> *out) {
  size_t start = 0, found = 0, end = full.size();
  out->clear();
  while (found != std::string::npos) {
    found = full.find_first_of(delim, start);
    // start != end condition is for when the delimiter is at the end
    if (!omit_empty_strings || (found != start && start != end))
      out->push_back(full.substr(start, found - start));
    start = found + 1;
  }
}

void ExtractHws(string hws_file, unordered_map<string, int> &hws_map)
{
    if(hws_file.empty()){
        return;
    }
    std::string line;
    std::ifstream ifs_hws(hws_file.c_str());
    if(!ifs_hws.is_open()){
        LOG(ERROR) << "Unable to open hotwords file: " << hws_file 
            << ". If you have not set hotwords, please ignore this message.";
        return;
    }
    LOG(INFO) << "hotwords: ";
    while (getline(ifs_hws, line)) {
        Trim(&line);
        if (line.empty()) {
            continue;
        }
        float score = 1.0f;
        std::vector<std::string> text;
        SplitStringToVector(line, " ", true, &text);
        
        if (text.size() > 1) {
            try{
                score = std::stof(text[text.size() - 1]);
            }catch (std::exception const &e)
            {
                LOG(ERROR)<<e.what();
                continue;
            }
        } else {
            continue;
        }
        std::string hotword = "";
        for (size_t i = 0; i < text.size()-1; ++i) {
            hotword = hotword + text[i];
            if(i != text.size()-2){
                hotword = hotword + " ";
            }
        }
        
        LOG(INFO) << hotword << " : " << score;
        hws_map.emplace(hotword, score);
    }
    ifs_hws.close();
}

void ExtractHws(string hws_file, unordered_map<string, int> &hws_map, string& nn_hotwords_)
{
    if(hws_file.empty()){
        return;
    }
    std::string line;
    std::ifstream ifs_hws(hws_file.c_str());
    if(!ifs_hws.is_open()){
        LOG(ERROR) << "Unable to open hotwords file: " << hws_file 
            << ". If you have not set hotwords, please ignore this message.";
        return;
    }
    LOG(INFO) << "hotwords: ";
    while (getline(ifs_hws, line)) {
        Trim(&line);
        if (line.empty()) {
            continue;
        }
        float score = 1.0f;
        std::vector<std::string> text;
        SplitStringToVector(line, " ", true, &text);
        
        if (text.size() > 1) {
            try{
                score = std::stof(text[text.size() - 1]);
            }catch (std::exception const &e)
            {
                LOG(ERROR)<<e.what();
                continue;
            }
        } else {
            continue;
        }
        std::string hotword = "";
        for (size_t i = 0; i < text.size()-1; ++i) {
            hotword = hotword + text[i];
            if(i != text.size()-2){
                hotword = hotword + " ";
            }
        }
        
        nn_hotwords_ += " " + hotword;
        LOG(INFO) << hotword << " : " << score;
        hws_map.emplace(hotword, score);
    }
    ifs_hws.close();
}

} // namespace funasr

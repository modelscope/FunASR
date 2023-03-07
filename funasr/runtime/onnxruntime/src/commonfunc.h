#pragma once 


typedef struct
{
    std::string msg;

}RPASR_RECOG_RESULT;


#ifdef _WIN32
#include <codecvt>



inline std::wstring string2wstring(const std::string& str, const std::string& locale)
{
    typedef std::codecvt_byname<wchar_t, char, std::mbstate_t> F;
    std::wstring_convert<F> strCnv(new F(locale));
    return strCnv.from_bytes(str);
}

inline std::wstring  strToWstr(std::string str) {
    if (str.length() == 0)
        return L"";
    return  string2wstring(str, "zh-CN");

}

#endif



inline void getInputName(Ort::Session* session, string& inputName,int nIndex=0) {
    size_t numInputNodes = session->GetInputCount();
    if (numInputNodes > 0) {
        Ort::AllocatorWithDefaultOptions allocator;
        {
            auto t = session->GetInputNameAllocated(nIndex, allocator);
            inputName = t.get();

        }
    }
}

inline void getOutputName(Ort::Session* session, string& outputName, int nIndex = 0) {
    size_t numOutputNodes = session->GetOutputCount();
    if (numOutputNodes > 0) {
        Ort::AllocatorWithDefaultOptions allocator;
        {
            auto t = session->GetOutputNameAllocated(nIndex, allocator);
            outputName = t.get();

        }
    }
}
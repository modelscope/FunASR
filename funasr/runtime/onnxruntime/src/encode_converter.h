/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/
#ifndef __WS__ENCODE_CONVERTER_H__
#define __WS__ENCODE_CONVERTER_H__

#include <string>
#include <stdint.h>
#include <vector>
#ifdef _MSC_VER
#include <windows.h>
#endif // _MSC_VER

namespace funasr {
    typedef unsigned char           U8CHAR_T;
    typedef unsigned short          U16CHAR_T;
    typedef std::basic_string<U8CHAR_T>  u8string;
    typedef std::basic_string<U16CHAR_T> u16string;

    class EncodeConverter {
    public:
        static const U16CHAR_T defUniChar = 0x25a1;  //WHITE SQUARE

    public:
        static void SwapEndian(U16CHAR_T* pbuf, size_t len);

        static size_t Utf16ToUtf8(const U16CHAR_T* pu16, U8CHAR_T* pu8);

        ///< @param pu16 UTF16 string
        ///< @param pu8 UTF8 string
        static size_t Utf16ToUtf8(const U16CHAR_T* pu16, size_t ilen,
                                  U8CHAR_T* pu8, size_t olen);

        static u8string Utf16ToUtf8(const u16string& u16str);

        static size_t Utf8ToUtf16(const U8CHAR_T* pu8, U16CHAR_T* pu16);

        static size_t Utf8ToUtf16(const U8CHAR_T* pu8, size_t ilen, U16CHAR_T* pu16);

        ///< @param pu8 UTF8 string
        ///< @param pu16 UTF16 string
        static size_t Utf8ToUtf16(const U8CHAR_T* pu8, size_t ilen,
                                  U16CHAR_T* pu16, size_t olen);

        static u16string Utf8ToUtf16(const u8string& u8str);

        ///< @param pu8 string
        ///< @return if string is encoded as UTF8 - true, otherwise false
        static bool IsUTF8(const U8CHAR_T* pu8, size_t ilen);

        ///< @param u8str string
        ///< @return if string is encoded as UTF8 - true, otherwise false
        static bool IsUTF8(const u8string& u8str);

        ///< @param UTF8 string
        ///< @return the word number of UTF8
        static size_t GetUTF8Len(const U8CHAR_T* pu8, size_t ilen);

        ///< @param UTF8 string
        ///< @return the word number of UTF8
        static size_t GetUTF8Len(const u8string& u8str);

        ///< @param pu16 UTF16 string
        ///< @param ilen UTF16 length
        ///< @return UTF8 string length
        static size_t Utf16ToUtf8Len(const U16CHAR_T* pu16, size_t ilen);

        static uint16_t ToUni(const char* sc, int &len);

        static bool IsChineseCharacter(U16CHAR_T &u16) {
            return (u16 >= 0x4e00 && u16 <= 0x9fff)  // common
                || (u16 >= 0x3400 && u16 <= 0x4dff); // rare, extension A
        }

        // whether the string is all Chinese
        static bool IsAllChineseCharactor(const U8CHAR_T* pu8, size_t ilen);
        static bool HasAlpha(const U8CHAR_T* pu8, size_t ilen);
        static bool NeedAddTailBlank(std::string str);
        static bool IsAllAlpha(const U8CHAR_T* pu8, size_t ilen);
        static bool IsAllAlphaAndPunct(const U8CHAR_T* pu8, size_t ilen);
        static bool IsAllAlphaAndDigit(const U8CHAR_T* pu8, size_t ilen);
        static bool IsAllAlphaAndDigitAndBlank(const U8CHAR_T* pu8, size_t ilen);
        static std::vector<std::string> MergeEnglishWord(std::vector<std::string> &str_vec_input,
                                                         std::vector<int> &merge_mask);
        static size_t Utf8ToCharset(const std::string &input, std::vector<std::string> &output);

#ifdef _MSC_VER
        // convert to the local ansi page
        static std::string UTF8ToLocaleAnsi(const std::string& strUTF8) {
            int len = MultiByteToWideChar(CP_UTF8, 0, strUTF8.c_str(), -1, NULL, 0);
            unsigned short*wszGBK = new unsigned short[len + 1];
            memset(wszGBK, 0, len * 2 + 2);
            MultiByteToWideChar(CP_UTF8, 0, (LPCCH)strUTF8.c_str(), -1, (LPWSTR)wszGBK, len);

            len = WideCharToMultiByte(CP_ACP, 0, (LPCWCH)wszGBK, -1, NULL, 0, NULL, NULL);
            char *szGBK = new char[len + 1];
            memset(szGBK, 0, len + 1);
            WideCharToMultiByte(CP_ACP, 0, (LPCWCH)wszGBK, -1, szGBK, len, NULL, NULL);
            std::string strTemp(szGBK);
            delete[]szGBK;
            delete[]wszGBK;
            return strTemp;
        }
#endif
    };
}

#endif //__WS_ENCODE_CONVERTER_H__

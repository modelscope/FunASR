/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/
#include "encode_converter.h"
#include <assert.h>


namespace funasr {
using namespace std;

U16CHAR_T UTF16[8];
U8CHAR_T UTF8[8];

size_t MyUtf8ToUtf16(const U8CHAR_T* pu8, size_t ilen, U16CHAR_T* pu16);
size_t MyUtf16ToUtf8(const U16CHAR_T* pu16, U8CHAR_T* pu8);


void EncodeConverter::SwapEndian(U16CHAR_T* pbuf, size_t len)
{
  for (size_t i = 0; i < len; i++) {
    pbuf[i] = ((pbuf[i] >> 8) | (pbuf[i] << 8));
  }
}


size_t MyUtf16ToUtf8(const U16CHAR_T* pu16, U8CHAR_T* pu8)
{
  size_t n = 0;
  if (pu16[0] <= 0x007F)
  {
    pu8[0] = (pu16[0] & 0x7F);
    n = 1;
  }
  else if (pu16[0] >= 0x0080 &&  pu16[0] <= 0x07FF)
  {
    pu8[1] = (0x80 | (pu16[0] & 0x003F));
    pu8[0] = (0xC0 | ((pu16[0] >> 6) & 0x001F));
    n = 2;
  }
  else if (pu16[0] >= 0x0800)
  {
    pu8[2] = (0x80 | (pu16[0] & 0x003F));
    pu8[1] = (0x80 | ((pu16[0] >> 6) & 0x003F));
    pu8[0] = (0xE0 | ((pu16[0] >> 12) & 0x000F));
    n = 3;
  }

  return n;
}

#define is2ByteUtf16(u16) ( (u16) >= 0x0080 && (u16) <= 0x07FF )
#define is3ByteUtf16(u16) ( (u16) >= 0x0800 )

size_t EncodeConverter::Utf16ToUtf8(const U16CHAR_T* pu16, U8CHAR_T* pu8)
{
  size_t n = 0;
  if (pu16[0] <= 0x007F)
  {
    pu8[0] = (pu16[0] & 0x7F);
    n = 1;
  }
  else if (pu16[0] >= 0x0080 &&  pu16[0] <= 0x07FF)
  {
    pu8[1] = (0x80 | (pu16[0] & 0x003F));
    pu8[0] = (0xC0 | ((pu16[0] >> 6) & 0x001F));
    n = 2;
  }
  else if (pu16[0] >= 0x0800)
  {
    pu8[2] = (0x80 | (pu16[0] & 0x003F));
    pu8[1] = (0x80 | ((pu16[0] >> 6) & 0x003F));
    pu8[0] = (0xE0 | ((pu16[0] >> 12) & 0x000F));
    n = 3;
  }

  return n;
}

size_t EncodeConverter::Utf16ToUtf8(const U16CHAR_T* pu16, size_t ilen,
    U8CHAR_T* pu8, size_t olen)
{
  size_t offset = 0;
  size_t sz = 0;
  /*
  for (size_t i = 0; i < ilen && offset < static_cast<int>(olen) - 3; i++) {
    sz = utf16ToUtf8(pu16 + i, pu8 + offset);
    offset += sz;
  }
  */
  for (size_t i = 0; i < ilen && static_cast<int>(offset) < static_cast<int>(olen); i++) {
    sz = Utf16ToUtf8(pu16 + i, pu8 + offset);
    if (static_cast<int>(offset + static_cast<int>(sz)) <= static_cast<int>(olen))
        offset += sz;
  }
  
 // pu8[offset] = '\0';
  return offset;
}

u8string EncodeConverter::Utf16ToUtf8(const u16string& u16str)
{
  size_t buflen = u16str.length()*3 + 1;
  U8CHAR_T* pu8 = new U8CHAR_T[buflen];
  size_t len = Utf16ToUtf8(u16str.data(), u16str.length(),
    pu8, buflen);
  u8string u8str(pu8, len);
  delete [] pu8;

  return u8str;
}

size_t EncodeConverter::Utf8ToUtf16(const U8CHAR_T* pu8, U16CHAR_T* pu16)
{
  size_t n = 0;
  if ((pu8[0] & 0xF0) == 0xE0)
  {
    if ((pu8[1] & 0xC0) == 0x80 &&
        (pu8[2] & 0xC0) == 0x80)
    {
      pu16[0] = (((pu8[0] & 0x0F) << 4) | ((pu8[1] & 0x3C) >> 2));
      pu16[0] <<= 8;
      pu16[0] |= (((pu8[1] & 0x03) << 6) | (pu8[2] & 0x3F));
    }
    else
    {
      pu16[0] = defUniChar;
    }
    n = 3;
  } 
  else if ((pu8[0] & 0xE0) == 0xC0)
  {
    if ((pu8[1] & 0xC0) == 0x80) 
    {
      pu16[0] = ((pu8[0] & 0x1C) >> 2);
      pu16[0] <<= 8;
      pu16[0] |= (((pu8[0] & 0x03) << 6) | (pu8[1] & 0x3F));
    }
    else
    {
      pu16[0] = defUniChar;
    }
    n = 2;
  } 
  else if ((pu8[0] & 0x80) == 0x00) 
  {
    pu16[0] = pu8[0];
    n = 1;
  }

  return n;
}

size_t MyUtf8ToUtf16(const U8CHAR_T* pu8, size_t ilen, U16CHAR_T* pu16)
{
  size_t n = 0;
  if ((pu8[0] & 0xF0) == 0xE0 && ilen >= 3)
  {
    if ((pu8[1] & 0xC0) == 0x80 &&
        (pu8[2] & 0xC0) == 0x80)
    {
      pu16[0] = (((pu8[0] & 0x0F) << 4) | ((pu8[1] & 0x3C) >> 2));
      pu16[0] <<= 8;
      pu16[0] |= (((pu8[1] & 0x03) << 6) | (pu8[2] & 0x3F));
      n = 3;
    }
    else
    {
      pu16[0] = 0x0000;
      n = 1;
    }
  } 
  else if ((pu8[0] & 0xE0) == 0xC0 && ilen >= 2)
  {
    if ((pu8[1] & 0xC0) == 0x80) 
    {
      pu16[0] = ((pu8[0] & 0x1C) >> 2);
      pu16[0] <<= 8;
      pu16[0] |= (((pu8[0] & 0x03) << 6) | (pu8[1] & 0x3F));
      n = 2;
    }
    else
    {
      pu16[0] = 0x0000;
      n = 1;
    }
  } 
  else if ((pu8[0] & 0x80) == 0x00) 
  {
    pu16[0] = pu8[0];
    n = 1;
  }
  else
  {
      pu16[0] = 0x0000;
      n = 1;
  }
  return n;
}

size_t EncodeConverter::Utf8ToUtf16(const U8CHAR_T* pu8, size_t ilen, U16CHAR_T* pu16)
{
  size_t n = 0;
  if ((pu8[0] & 0xF0) == 0xE0 && ilen >= 3)
  {
    if ((pu8[1] & 0xC0) == 0x80 &&
        (pu8[2] & 0xC0) == 0x80)
    {
      pu16[0] = (((pu8[0] & 0x0F) << 4) | ((pu8[1] & 0x3C) >> 2));
      pu16[0] <<= 8;
      pu16[0] |= (((pu8[1] & 0x03) << 6) | (pu8[2] & 0x3F));
      n = 3;
      if( !is3ByteUtf16(pu16[0]) )
      {
          pu16[0] = 0x0000;
          n = 1;
      }
    }
    else
    {
      pu16[0] = 0x0000;
      n = 1;
    }
  } 
  else if ((pu8[0] & 0xE0) == 0xC0 && ilen >= 2)
  {
    if ((pu8[1] & 0xC0) == 0x80) 
    {
      pu16[0] = ((pu8[0] & 0x1C) >> 2);
      pu16[0] <<= 8;
      pu16[0] |= (((pu8[0] & 0x03) << 6) | (pu8[1] & 0x3F));
      n = 2;
      if( !is2ByteUtf16(pu16[0]) )
      {
          pu16[0] = 0x0000;
          n = 1;
      }
    }
    else
    {
      pu16[0] = 0x0000;
      n = 1;
    }
  } 
  else if ((pu8[0] & 0x80) == 0x00) 
  {
    pu16[0] = pu8[0];
    n = 1;
  }
  else
  {
      pu16[0] = 0x0000;
      n = 1;
  }

  return n;
  /*
  size_t n = 0;
  if ((pu8[0] & 0xF0) == 0xE0)
  {
    if (ilen >= 3 && (pu8[1] & 0xC0) == 0x80 &&
        (pu8[2] & 0xC0) == 0x80)
    {
      pu16[0] = (((pu8[0] & 0x0F) << 4) | ((pu8[1] & 0x3C) >> 2));
      pu16[0] <<= 8;
      pu16[0] |= (((pu8[1] & 0x03) << 6) | (pu8[2] & 0x3F));
    }
    else
    {
      pu16[0] = defUniChar;
    }
    n = 3;
  } 
  else if ((pu8[0] & 0xE0) == 0xC0)
  {
    if( ilen >= 2 && (pu8[1] & 0xC0) == 0x80) 
    {
      pu16[0] = ((pu8[0] & 0x1C) >> 2);
      pu16[0] <<= 8;
      pu16[0] |= (((pu8[0] & 0x03) << 6) | (pu8[1] & 0x3F));
    }
    else
    {
      pu16[0] = defUniChar;
    }
    n = 2;
  } 
  else if ((pu8[0] & 0x80) == 0x00) 
  {
    pu16[0] = pu8[0];
    n = 1;
  }
  else
  {
      pu16[0] = defUniChar;
      n = 1;
      for (size_t i = 1; i < ilen; i++)
      {
          if ((pu8[i] & 0xF0) == 0xE0 || (pu8[i] & 0xE0) == 0xC0 || (pu8[i] & 0x80) == 0x00)
              break;
          n++;
      }
  }

  return n;
  */
}

size_t EncodeConverter::Utf8ToUtf16(const U8CHAR_T* pu8, size_t ilen,
    U16CHAR_T* pu16, size_t olen)
{
  int offset = 0;
  size_t sz = 0;
  for (size_t i = 0; i < ilen && offset < static_cast<int>(olen); offset ++)
  {
    sz = Utf8ToUtf16(pu8 + i, ilen - i, pu16 + offset);
    i += sz;
    if (sz == 0) {
      // failed
      // assert(sz != 0);
      break;
    }
  }
//  pu16[offset] = '\0';

  return offset;
}

u16string EncodeConverter::Utf8ToUtf16(const u8string& u8str)
{
  U16CHAR_T* p16 = new U16CHAR_T[u8str.length() + 1];
  size_t len = Utf8ToUtf16(u8str.data(), u8str.length(), 
      p16, u8str.length() + 1);
  u16string u16str(p16, len);
  delete[] p16;

  return u16str;
}

bool EncodeConverter::IsUTF8(const U8CHAR_T* pu8, size_t ilen)
{
  size_t i;
  size_t n = 0;
  for (i = 0; i < ilen; i += n)
  {
    if ((pu8[i] & 0xF0) == 0xE0 &&
        (pu8[i + 1] & 0xC0) == 0x80 &&
        (pu8[i + 2] & 0xC0) == 0x80)
    {
      n = 3;
    }
    else if ((pu8[i] & 0xE0) == 0xC0 &&
        (pu8[i + 1] & 0xC0) == 0x80)
    {
      n = 2;
    }
    else if ((pu8[i] & 0x80) == 0x00)
    {
      n = 1;
    }
    else
    {
      break;
    }
  }

  return i == ilen;
}

bool EncodeConverter::IsUTF8(const u8string& u8str)
{
  return IsUTF8(u8str.data(), u8str.length());
}
  
size_t EncodeConverter::GetUTF8Len(const U8CHAR_T* pu8, size_t ilen)
{
  size_t i;
  size_t n = 0;
  size_t rlen = 0;
  for (i = 0; i < ilen; i += n, rlen ++)
  {
    if ((pu8[i] & 0xF0) == 0xE0 &&
        (pu8[i + 1] & 0xC0) == 0x80 &&
        (pu8[i + 2] & 0xC0) == 0x80)
    {
      n = 3;
    }
    else if ((pu8[i] & 0xE0) == 0xC0 &&
        (pu8[i + 1] & 0xC0) == 0x80)
    {
      n = 2;
    }
    else if ((pu8[i] & 0x80) == 0x00)
    {
      n = 1;
    }
    else
    {
      break;
    }
  }

  if (i == ilen)
    return 0;
  else
    return rlen;
}

size_t EncodeConverter::GetUTF8Len(const u8string& u8str)
{
  return GetUTF8Len(u8str.data(), u8str.length());
}


size_t EncodeConverter::Utf16ToUtf8Len(const U16CHAR_T* pu16, size_t ilen)
{
  int offset = 0;
  for (size_t i = 0; i < ilen ; i++) {
      if (pu16[i] <= 0x007F)
      {
        offset += 1;
      }
      else if (pu16[i] >= 0x0080 &&  pu16[i] <= 0x07FF)
      {
        offset += 2;
      }
      else if (pu16[i] >= 0x0800)
      {
        offset += 3;
      }
  }
  
  return offset;
}

uint16_t EncodeConverter::ToUni(const char* sc, int &len)
{
    uint16_t wide[2];
    len = (int)Utf8ToUtf16((const U8CHAR_T*)sc, wide);
    return wide[0];
}

bool EncodeConverter::IsAllChineseCharactor(const U8CHAR_T* pu8, size_t ilen) {
    if (pu8 == NULL || ilen <= 0) {
        return false;
    }

    U16CHAR_T* p16 = new U16CHAR_T[ilen + 1];
    size_t len = Utf8ToUtf16(pu8, ilen, p16, ilen + 1);
    for (size_t i = 0; i < len; i++) {
        if (p16[i] < 0x4e00 || p16[i] > 0x9fff) {
            delete[] p16;
            return false;
        }
    }
    delete[] p16;
    return true;
}

bool EncodeConverter::HasAlpha(const U8CHAR_T* pu8, size_t ilen) {
  if (pu8 == NULL || ilen <= 0) {
    return false;
  }
  for (size_t i = 0; i < ilen; i++) {
    if (pu8[i]> 0 && isalpha(pu8[i])){
      return true;
    }
  }
  return false;
}


bool EncodeConverter::IsAllAlpha(const U8CHAR_T* pu8, size_t ilen) {
  if (pu8 == NULL || ilen <= 0) {
    return false;
  }
  for (size_t i = 0; i < ilen; i++) {
    if (!(pu8[i]> 0 && isalpha(pu8[i]))){
      return false;
    }
  }
  return true;
}

bool EncodeConverter::IsAllAlphaAndPunct(const U8CHAR_T* pu8, size_t ilen) {
  if (pu8 == NULL || ilen <= 0) {
    return false;
  }
  bool flag1 = HasAlpha(pu8, ilen);
  if (flag1 == false) {
    return false;
  }

  for (size_t i = 0; i < ilen; i++) {
    if (!(pu8[i]> 0 && (isalpha(pu8[i]) || (ispunct(pu8[i]))))){
      return false;
    }
  }
  return true;
}

bool EncodeConverter::IsAllAlphaAndDigit(const U8CHAR_T* pu8, size_t ilen) {
  if (pu8 == NULL || ilen <= 0) {
    return false;
  }
  bool flag1 = HasAlpha(pu8, ilen);
  if (flag1 == false) {
    return false;
  }

  for (size_t i = 0; i < ilen; i++) {
    if (!(pu8[i]> 0 && (isalnum(pu8[i]) || isalpha(pu8[i]) || pu8[i] == '\''))){
      return false;
    }
  }
  return true;
}
bool EncodeConverter::IsAllAlphaAndDigitAndBlank(const U8CHAR_T* pu8, size_t ilen) {
  if (pu8 == NULL || ilen <= 0) {
    return false;
  }
  for (size_t i = 0; i < ilen; i++) {
    if (!(pu8[i]> 0 && (isalnum(pu8[i]) || isalpha(pu8[i]) || isblank(pu8[i]) || pu8[i] == '\''))){
      return false;
    }
  }
  return true;
}
bool EncodeConverter::NeedAddTailBlank(std::string str) {
  U8CHAR_T *pu8 = (U8CHAR_T*)str.data();
  size_t ilen = str.size();
  if (pu8 == NULL || ilen <= 0) {
    return false;
  }
  if (IsAllAlpha(pu8, ilen) || IsAllAlphaAndPunct(pu8, ilen) || IsAllAlphaAndDigit(pu8, ilen)) {
    return true;
  } else {
    return false;
  }
}
std::vector<std::string> EncodeConverter::MergeEnglishWord(std::vector<std::string> &str_vec_input,
                                                           std::vector<int> &merge_mask) {
  std::vector<std::string> output;
  for (int i = 0; i < merge_mask.size(); i++) {
    if (merge_mask[i] == 1 && i > 0) {
      output[output.size() - 1] += str_vec_input[i];
    } else {
      output.push_back(str_vec_input[i]);
    }
  }
  str_vec_input.swap(output);
  return str_vec_input;
}
size_t EncodeConverter::Utf8ToCharset(const std::string &input, std::vector<std::string> &output) {
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
}

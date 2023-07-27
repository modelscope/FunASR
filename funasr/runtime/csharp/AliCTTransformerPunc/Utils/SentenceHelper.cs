using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AliCTTransformerPunc.Utils
{
    /// <summary>
    /// SentenceHelper
    /// Copyright (c)  2023 by manyeyes
    /// </summary>
    internal class SentenceHelper
    {
        public static string[] CodeMixSplitWords(string text)
        {
            List<string> words = new List<string>();
            string[] segs = text.Split();
            foreach (var seg in segs)
            {
                //There is no space in seg.
                string current_word = "";
                foreach (var c in seg)
                {
                    if (c <= sbyte.MaxValue)
                    {
                        //This is an ASCII char.
                        current_word += c;
                    }
                    else
                    {
                        // This is a Chinese char.
                        if (current_word.Length > 0)
                        {
                            words.Add(current_word);
                            current_word = "";
                        }
                        words.Add(c.ToString());
                    }
                }
                if (current_word.Length > 0)
                {
                    words.Append(current_word);
                }
            }
            return words.ToArray();
        }

        public static int[] Tokens2ids(string[] _tokens, string[] splitText)
        {
            int[] ids = new int[splitText.Length];
            for (int i = 0; i < splitText.Length; i++)
            {
                ids[i] = Array.IndexOf(_tokens, splitText[i]);
            }
            return ids;
        }

        public static List<T[]> SplitToMiniSentence<T>(T[] words, int wordLimit = 20)
        {
            List<T[]> wordsList = new List<T[]>();
            if (words.Length <= wordLimit)
            {
                wordsList.Add(words);
            }
            else
            {
                string[] sentences;
                int length = words.Length;
                int sentenceLen = length / wordLimit;
                for (int i = 0; i < sentenceLen; i++)
                {
                    T[] vs = new T[wordLimit];
                    Array.Copy(words, i * wordLimit, vs, 0, wordLimit);
                    wordsList.Add(vs);
                }
                int tailLength = length % wordLimit;
                if (tailLength > 0)
                {
                    T[] vs = new T[wordLimit];
                    Array.Copy(words, sentenceLen * wordLimit, vs, 0, tailLength);
                    wordsList.Add(vs);
                }
            }
            return wordsList;
        }
    }
}

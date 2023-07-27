// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AliParaformerAsr.Model;
using AliParaformerAsr.Utils;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Extensions.Logging;
using System.Text.RegularExpressions;

namespace AliParaformerAsr
{
    /// <summary>
    /// offline recognizer package
    /// Copyright (c)  2023 by manyeyes
    /// </summary>
    public class OfflineRecognizer
    {
        private InferenceSession _onnxSession;
        private readonly ILogger<OfflineRecognizer> _logger;
        private WavFrontend _wavFrontend;
        private string _frontend;
        private FrontendConfEntity _frontendConfEntity;
        private string[] _tokens;
        private int _batchSize = 1;

        public OfflineRecognizer(string modelFilePath, string configFilePath, string mvnFilePath,string tokensFilePath, int batchSize = 1,int threadsNum=1)
        {
            Microsoft.ML.OnnxRuntime.SessionOptions options = new Microsoft.ML.OnnxRuntime.SessionOptions();
            //options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO;
            //options.AppendExecutionProvider_DML(0);
            options.AppendExecutionProvider_CPU(0);
            //options.AppendExecutionProvider_CUDA(0);
            options.InterOpNumThreads = threadsNum;
            _onnxSession = new InferenceSession(modelFilePath, options);

            _tokens = File.ReadAllLines(tokensFilePath);

            OfflineYamlEntity offlineYamlEntity = YamlHelper.ReadYaml<OfflineYamlEntity>(configFilePath);
            _wavFrontend = new WavFrontend(mvnFilePath, offlineYamlEntity.frontend_conf);
            _frontend = offlineYamlEntity.frontend;
            _frontendConfEntity = offlineYamlEntity.frontend_conf;
            _batchSize = batchSize;
            ILoggerFactory loggerFactory = new LoggerFactory();
            _logger = new Logger<OfflineRecognizer>(loggerFactory);
        }

        public List<string> GetResults(List<float[]> samples)
        {
            this._logger.LogInformation("get features begin");
            List<OfflineInputEntity> offlineInputEntities = ExtractFeats(samples);
            OfflineOutputEntity modelOutput = this.Forward(offlineInputEntities);
            List<string> text_results = this.DecodeMulti(modelOutput.Token_nums);
            return text_results;
        }

        private List<OfflineInputEntity> ExtractFeats(List<float[]> waveform_list)
        {
            List<float[]> in_cache = new List<float[]>();
            List<OfflineInputEntity> offlineInputEntities = new List<OfflineInputEntity>();
            foreach (var waveform in waveform_list)
            {
                float[] fbanks = _wavFrontend.GetFbank(waveform);
                float[] features = _wavFrontend.LfrCmvn(fbanks);
                OfflineInputEntity offlineInputEntity = new OfflineInputEntity();
                offlineInputEntity.Speech = features;
                offlineInputEntity.SpeechLength = features.Length;
                offlineInputEntities.Add(offlineInputEntity);
            }
            return offlineInputEntities;
        }

        private OfflineOutputEntity Forward(List<OfflineInputEntity> modelInputs)
        {
            int BatchSize = modelInputs.Count;
            float[] padSequence = PadSequence(modelInputs);
            var inputMeta = _onnxSession.InputMetadata;
            var container = new List<NamedOnnxValue>(); 
            foreach (var name in inputMeta.Keys)
            {
                if (name == "speech")
                {
                    int[] dim = new int[] { BatchSize, padSequence.Length / 560 / BatchSize, 560 };
                    var tensor = new DenseTensor<float>(padSequence, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
                if (name == "speech_lengths")
                {
                    int[] dim = new int[] { BatchSize };
                    int[] speech_lengths = new int[BatchSize];
                    for (int i = 0; i < BatchSize; i++)
                    {
                        speech_lengths[i] = padSequence.Length / 560 / BatchSize;
                    }
                    var tensor = new DenseTensor<int>(speech_lengths, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<int>(name, tensor));
                }
            }


            IReadOnlyCollection<string> outputNames = new List<string>();
            outputNames.Append("logits");
            outputNames.Append("token_num");
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = null;
            try
            {
                results = _onnxSession.Run(container);
            }
            catch (Exception ex)
            {
                //
            }
            OfflineOutputEntity modelOutput = new OfflineOutputEntity();
            if (results != null)
            {
                var resultsArray = results.ToArray();
                modelOutput.Logits = resultsArray[0].AsEnumerable<float>().ToArray();
                modelOutput.Token_nums_length = resultsArray[1].AsEnumerable<int>().ToArray();

                Tensor<float> logits_tensor = resultsArray[0].AsTensor<float>();
                Tensor<Int64> token_nums_tensor = resultsArray[1].AsTensor<Int64>();

                List<int[]> token_nums = new List<int[]> { };

                for (int i = 0; i < logits_tensor.Dimensions[0]; i++)
                {
                    int[] item = new int[logits_tensor.Dimensions[1]];
                    for (int j = 0; j < logits_tensor.Dimensions[1]; j++)
                    {                        
                        int token_num = 0;
                        for (int k = 1; k < logits_tensor.Dimensions[2]; k++)
                        {
                            token_num = logits_tensor[i, j, token_num] > logits_tensor[i, j, k] ? token_num : k;
                        }
                        item[j] = (int)token_num;                        
                    }
                    token_nums.Add(item);
                }                
                modelOutput.Token_nums = token_nums;
            }
            return modelOutput;
        }

        private List<string> DecodeMulti(List<int[]> token_nums)
        {
            List<string> text_results = new List<string>();
#pragma warning disable CS8602 // 解引用可能出现空引用。
            foreach (int[] token_num in token_nums)
            {
                string text_result = "";
                foreach (int token in token_num)
                {
                    if (token == 2)
                    {
                        break;
                    }
                    if (_tokens[token] != "</s>" && _tokens[token] != "<s>" && _tokens[token] != "<blank>")
                    {                        
                        if (IsChinese(_tokens[token],true))
                        {
                            text_result += _tokens[token];
                        }
                        else
                        {
                            text_result += "▁" + _tokens[token]+ "▁";
                        }
                    }
                }
                text_results.Add(text_result.Replace("@@▁▁", "").Replace("▁▁", " ").Replace("▁", ""));
            }
#pragma warning restore CS8602 // 解引用可能出现空引用。

            return text_results;
        }

        /// <summary>
        /// Verify if the string is in Chinese.
        /// </summary>
        /// <param name="checkedStr">The string to be verified.</param>
        /// <param name="allMatch">Is it an exact match. When the value is true,all are in Chinese; 
        /// When the value is false, only Chinese is included.
        /// </param>
        /// <returns></returns>
        private bool IsChinese(string checkedStr, bool allMatch)
        {
            string pattern;
            if (allMatch)
                pattern = @"^[\u4e00-\u9fa5]+$";
            else
                pattern = @"[\u4e00-\u9fa5]";
            if (Regex.IsMatch(checkedStr, pattern))
                return true;
            else
                return false;
        }

        private float[] PadSequence(List<OfflineInputEntity> modelInputs)
        {
            int max_speech_length = modelInputs.Max(x => x.SpeechLength);
            int speech_length = max_speech_length * modelInputs.Count;
            float[] speech = new float[speech_length];
            float[,] xxx = new float[modelInputs.Count, max_speech_length];
            for (int i = 0; i < modelInputs.Count; i++)
            {
                if (max_speech_length == modelInputs[i].SpeechLength)
                {
                    for (int j = 0; j < xxx.GetLength(1); j++)
                    {
#pragma warning disable CS8602 // 解引用可能出现空引用。
                        xxx[i, j] = modelInputs[i].Speech[j];
#pragma warning restore CS8602 // 解引用可能出现空引用。
                    }
                    continue;
                }
                float[] nullspeech = new float[max_speech_length - modelInputs[i].SpeechLength];
                float[]? curr_speech = modelInputs[i].Speech;
                float[] padspeech = new float[max_speech_length];
                padspeech = _wavFrontend.ApplyCmvn(padspeech);
                Array.Copy(curr_speech, 0, padspeech, 0, curr_speech.Length);
                for (int j = 0; j < padspeech.Length; j++)
                {
#pragma warning disable CS8602 // 解引用可能出现空引用。
                    xxx[i, j] = padspeech[j];
#pragma warning restore CS8602 // 解引用可能出现空引用。
                }

            }
            int s = 0;
            for (int i = 0; i < xxx.GetLength(0); i++)
            {
                for (int j = 0; j < xxx.GetLength(1); j++)
                {
                    speech[s] = xxx[i, j];
                    s++;
                }
            }
            return speech;
        }
    }
}
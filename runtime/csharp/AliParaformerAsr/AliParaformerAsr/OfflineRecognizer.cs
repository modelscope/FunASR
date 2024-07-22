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
using Newtonsoft.Json.Linq;

// 模型文件地址： https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx
namespace AliParaformerAsr
{
    public enum OnnxRumtimeTypes
    {
        CPU = 0,

        DML = 1,

        CUDA = 2,
    }

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

        /// <summary>
        /// 
        /// </summary>
        /// <param name="modelFilePath"></param>
        /// <param name="configFilePath"></param>
        /// <param name="mvnFilePath"></param>
        /// <param name="tokensFilePath"></param>
        /// <param name="rumtimeType">可以选择gpu，但是目前情况下，不建议使用，因为性能提升有限</param>
        /// <param name="deviceId">设备id，多显卡时用于指定执行的显卡</param>
        /// <param name="batchSize"></param>
        /// <param name="threadsNum"></param>
        /// <exception cref="ArgumentException"></exception>
        public OfflineRecognizer(string modelFilePath, string configFilePath, string mvnFilePath, string tokensFilePath, OnnxRumtimeTypes rumtimeType = OnnxRumtimeTypes.CPU, int deviceId = 0)
        {
            var options = new SessionOptions();
            switch(rumtimeType)
            {
                case OnnxRumtimeTypes.DML:
                    options.AppendExecutionProvider_DML(deviceId);
                    break;
                case OnnxRumtimeTypes.CUDA:
                    options.AppendExecutionProvider_CUDA(deviceId);
                    break;
                default:
                    options.AppendExecutionProvider_CPU(deviceId);
                    break;
            }
            //options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO;

            _onnxSession = new InferenceSession(modelFilePath, options);
            
            string[] tokenLines;
            if (tokensFilePath.EndsWith(".txt"))
            {
                tokenLines = File.ReadAllLines(tokensFilePath);
            }
            else if (tokensFilePath.EndsWith(".json"))
            {
                string jsonContent = File.ReadAllText(tokensFilePath);
                JArray tokenArray = JArray.Parse(jsonContent);
                tokenLines = tokenArray.Select(t => t.ToString()).ToArray();
            }
            else
            {
                throw new ArgumentException("Invalid tokens file format. Only .txt and .json are supported.");
            }

            _tokens = tokenLines;

            OfflineYamlEntity offlineYamlEntity = YamlHelper.ReadYaml<OfflineYamlEntity>(configFilePath);
            _wavFrontend = new WavFrontend(mvnFilePath, offlineYamlEntity.frontend_conf);
            _frontend = offlineYamlEntity.frontend;
            _frontendConfEntity = offlineYamlEntity.frontend_conf;
            ILoggerFactory loggerFactory = new LoggerFactory();
            _logger = new Logger<OfflineRecognizer>(loggerFactory);
        }

        public List<string> GetResults(List<float[]> samples)
        {
            _logger.LogInformation("get features begin");
            List<OfflineInputEntity> offlineInputEntities = ExtractFeats(samples);
            OfflineOutputEntity modelOutput = Forward(offlineInputEntities);
            List<string> text_results = DecodeMulti(modelOutput.Token_nums);
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
                    int[] dim = new int[] { BatchSize, padSequence.Length / 560 / BatchSize, 560 };//inputMeta["speech"].Dimensions[2]
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

                    string tokenChar = _tokens[token];

                    if (tokenChar != "</s>" && tokenChar != "<s>" && tokenChar != "<blank>")
                    {                        
                        if (IsChinese(tokenChar, true))
                        {
                            text_result += tokenChar;
                        }
                        else
                        {
                            text_result += "▁" + tokenChar + "▁";
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
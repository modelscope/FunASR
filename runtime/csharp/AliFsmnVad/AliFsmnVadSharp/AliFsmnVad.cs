using AliFsmnVadSharp.Model;
using AliFsmnVadSharp.Utils;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

// 模型文件下载地址：https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-onnx/

namespace AliFsmnVadSharp
{
    public class AliFsmnVad : IDisposable
    {
        private bool _disposed;
        private InferenceSession _onnxSession;
        private readonly ILogger _logger;
        private string _frontend;
        private WavFrontend _wavFrontend;
        private int _batchSize = 1;
        private int _max_end_sil = int.MinValue;
        private EncoderConfEntity _encoderConfEntity;
        private VadPostConfEntity _vad_post_conf;

        public AliFsmnVad(string modelFilePath, string configFilePath, string mvnFilePath, int batchSize = 1)
        {
            SessionOptions options = new SessionOptions();
            options.AppendExecutionProvider_CPU(0);
            options.InterOpNumThreads = 1;
            _onnxSession = new InferenceSession(modelFilePath, options);

            VadYamlEntity vadYamlEntity = YamlHelper.ReadYaml<VadYamlEntity>(configFilePath);
            _wavFrontend = new WavFrontend(mvnFilePath, vadYamlEntity.frontend_conf);
            _frontend = vadYamlEntity.frontend;
            _vad_post_conf = vadYamlEntity.model_conf;
            _batchSize = batchSize;
            _max_end_sil = _max_end_sil != int.MinValue ? _max_end_sil : vadYamlEntity.model_conf.max_end_silence_time;
            _encoderConfEntity = vadYamlEntity.encoder_conf;

            ILoggerFactory loggerFactory = new LoggerFactory();
            _logger = new Logger<AliFsmnVad>(loggerFactory);
        }

        public SegmentEntity[] GetSegments(List<float[]> samples)
        {
            int waveform_nums = samples.Count;
            _batchSize = Math.Min(waveform_nums, _batchSize);
            SegmentEntity[] segments = new SegmentEntity[waveform_nums];
            for (int beg_idx = 0; beg_idx < waveform_nums; beg_idx += _batchSize)
            {
                int end_idx = Math.Min(waveform_nums, beg_idx + _batchSize);
                List<float[]> waveform_list = new List<float[]>();
                for (int i = beg_idx; i < end_idx; i++)
                {
                    waveform_list.Add(samples[i]);
                }
                List<VadInputEntity> vadInputEntitys = ExtractFeats(waveform_list);
                try
                {
                    int t_offset = 0;
                    int step = Math.Min(waveform_list.Max(x => x.Length), 6000);
                    bool is_final = true;
                    List<VadOutputEntity> vadOutputEntitys = Infer(vadInputEntitys);
                    for (int batch_num = beg_idx; batch_num < end_idx; batch_num++)
                    {
                        var scores = vadOutputEntitys[batch_num - beg_idx].Scores;
                        SegmentEntity[] segments_part = vadInputEntitys[batch_num].VadScorer.DefaultCall(scores, waveform_list[batch_num - beg_idx], is_final: is_final, max_end_sil: _max_end_sil, online: false);
                        if (segments_part.Length > 0)
                        {
#pragma warning disable CS8602 // 解引用可能出现空引用。
                            if (segments[batch_num] == null)
                            {
                                segments[batch_num] = new SegmentEntity();
                            }
                            segments[batch_num].Segment.AddRange(segments_part[0].Segment); //
#pragma warning restore CS8602 // 解引用可能出现空引用。

                        }
                    }
                }
                catch (OnnxRuntimeException ex)
                {
                    _logger.LogWarning("input wav is silence or noise");
                    segments = null;
                }
//                for (int batch_num = 0; batch_num < _batchSize; batch_num++)
//                {
//                    List<float[]> segment_waveforms = new List<float[]>();
//                    foreach (int[] segment in segments[beg_idx + batch_num].Segment)
//                    {
//                        // (int)(16000 * (segment[0] / 1000.0) * 2);
//                        int frame_length = (((6000 * 400) / 400 - 1) * 160 + 400) / 60 / 1000;
//                        int frame_start = segment[0] * frame_length;
//                        int frame_end = segment[1] * frame_length;
//                        float[] segment_waveform = new float[frame_end - frame_start];
//                        Array.Copy(waveform_list[batch_num], frame_start, segment_waveform, 0, segment_waveform.Length);
//                        segment_waveforms.Add(segment_waveform);
//                    }
//                    segments[beg_idx + batch_num].Waveform.AddRange(segment_waveforms);
//                }
            }

            return segments;
        }

        public SegmentEntity[] GetSegmentsByStep(List<float[]> samples)
        {
            int waveform_nums = samples.Count;
            _batchSize=Math.Min(waveform_nums, _batchSize);
            SegmentEntity[] segments = new SegmentEntity[waveform_nums];
            for (int beg_idx = 0; beg_idx < waveform_nums; beg_idx += _batchSize)
            {
                int end_idx = Math.Min(waveform_nums, beg_idx + _batchSize);
                List<float[]> waveform_list = new List<float[]>();
                for (int i = beg_idx; i < end_idx; i++)
                {
                    waveform_list.Add(samples[i]);
                }
                List<VadInputEntity> vadInputEntitys = ExtractFeats(waveform_list);
                int feats_len = vadInputEntitys.Max(x => x.SpeechLength);
                List<float[]> in_cache = new List<float[]>();
                in_cache = PrepareCache(in_cache);
                try
                {
                    int step = Math.Min(vadInputEntitys.Max(x => x.SpeechLength), 6000 * 400);
                    bool is_final = true;
                    for (int t_offset = 0; t_offset < (int)(feats_len); t_offset += Math.Min(step, feats_len - t_offset))
                    {

                        if (t_offset + step >= feats_len - 1)
                        {
                            step = feats_len - t_offset;
                            is_final = true;
                        }
                        else
                        {
                            is_final = false;
                        }
                        List<VadInputEntity> vadInputEntitys_step = new List<VadInputEntity>();
                        foreach (VadInputEntity vadInputEntity in vadInputEntitys)
                        {
                            VadInputEntity vadInputEntity_step = new VadInputEntity();
                            float[]? feats = vadInputEntity.Speech;
                            int curr_step = Math.Min(feats.Length - t_offset, step);
                            if (curr_step <= 0)
                            {
                                vadInputEntity_step.Speech = new float[32000];
                                vadInputEntity_step.SpeechLength = 0;
                                vadInputEntity_step.InCaches = in_cache;
                                vadInputEntity_step.Waveform = new float[(((int)(32000) / 400 - 1) * 160 + 400)];
                                vadInputEntitys_step.Add(vadInputEntity_step);
                                continue;
                            }
                            float[]? feats_step = new float[curr_step];
                            Array.Copy(feats, t_offset, feats_step, 0, feats_step.Length);
                            float[]? waveform = vadInputEntity.Waveform;
                            float[]? waveform_step = new float[Math.Min(waveform.Length, ((int)(t_offset + step) / 400 - 1) * 160 + 400) - t_offset / 400 * 160];
                            Array.Copy(waveform, t_offset / 400 * 160, waveform_step, 0, waveform_step.Length);
                            vadInputEntity_step.Speech = feats_step;
                            vadInputEntity_step.SpeechLength = feats_step.Length;
                            vadInputEntity_step.InCaches = vadInputEntity.InCaches;
                            vadInputEntity_step.Waveform = waveform_step;
                            vadInputEntitys_step.Add(vadInputEntity_step);
                        }
                        List<VadOutputEntity> vadOutputEntitys = Infer(vadInputEntitys_step);
                        for (int batch_num = 0; batch_num < _batchSize; batch_num++)
                        {
                            vadInputEntitys[batch_num].InCaches = vadOutputEntitys[batch_num].OutCaches;
                            var scores = vadOutputEntitys[batch_num].Scores;
                            SegmentEntity[] segments_part = vadInputEntitys[batch_num].VadScorer.DefaultCall(scores, vadInputEntitys_step[batch_num].Waveform, is_final: is_final, max_end_sil: _max_end_sil, online: false);
                            if (segments_part.Length > 0)
                            {

#pragma warning disable CS8602 // 解引用可能出现空引用。
                                if (segments[beg_idx + batch_num] == null)
                                {
                                    segments[beg_idx + batch_num] = new SegmentEntity();
                                }
                                if (segments_part[0] != null)
                                {
                                    segments[beg_idx + batch_num].Segment.AddRange(segments_part[0].Segment);
                                }
#pragma warning restore CS8602 // 解引用可能出现空引用。

                            }
                        }
                    }
                }
                catch (OnnxRuntimeException ex)
                {
                    _logger.LogWarning("input wav is silence or noise");
                    segments = null;
                }
//                for (int batch_num = 0; batch_num < _batchSize; batch_num++)
//                {
//                    List<float[]> segment_waveforms=new List<float[]>();
//                    foreach (int[] segment in segments[beg_idx + batch_num].Segment)
//                    {
//                        // (int)(16000 * (segment[0] / 1000.0) * 2);
//                        int frame_length = (((6000 * 400) / 400 - 1) * 160 + 400) / 60 / 1000;
//                        int frame_start = segment[0] * frame_length;
//                        int frame_end = segment[1] * frame_length;
//                        if(frame_end > waveform_list[batch_num].Length)
//                        {
//                            break;
//                        }
//                        float[] segment_waveform = new float[frame_end - frame_start];
//                        Array.Copy(waveform_list[batch_num], frame_start, segment_waveform, 0, segment_waveform.Length);
//                        segment_waveforms.Add(segment_waveform);
//                    }
//                    segments[beg_idx + batch_num].Waveform.AddRange(segment_waveforms);
//                }

            }
            return segments;
        }

        private List<float[]> PrepareCache(List<float[]> in_cache)
        {
            if (in_cache.Count > 0)
            {
                return in_cache;
            }

            int fsmn_layers = _encoderConfEntity.fsmn_layers;

            int proj_dim = _encoderConfEntity.proj_dim;
            int lorder = _encoderConfEntity.lorder;

            for (int i = 0; i < fsmn_layers; i++)
            {
                float[] cache = new float[1 * proj_dim * (lorder - 1) * 1];
                in_cache.Add(cache);
            }
            return in_cache;
        }

        private List<VadInputEntity> ExtractFeats(List<float[]> waveform_list)
        {
            List<float[]> in_cache = new List<float[]>();
            in_cache = PrepareCache(in_cache);
            List<VadInputEntity> vadInputEntitys = new List<VadInputEntity>();
            foreach (var waveform in waveform_list)
            {
                float[] fbanks = _wavFrontend.GetFbank(waveform);
                float[] features = _wavFrontend.LfrCmvn(fbanks);
                VadInputEntity vadInputEntity = new VadInputEntity();
                vadInputEntity.Waveform = waveform;
                vadInputEntity.Speech = features;
                vadInputEntity.SpeechLength = features.Length;
                vadInputEntity.InCaches = in_cache;
                vadInputEntity.VadScorer = new E2EVadModel(_vad_post_conf);
                vadInputEntitys.Add(vadInputEntity);
            }
            return vadInputEntitys;
        }
        /// <summary>
        /// 一维数组转3维数组
        /// </summary>
        /// <param name="obj"></param>
        /// <param name="len">一维长</param>
        /// <param name="wid">二维长</param>
        /// <returns></returns>
        public static T[,,] DimOneToThree<T>(T[] oneDimObj, int len, int wid)
        {
            if (oneDimObj.Length % (len * wid) != 0)
                return null;
            int height = oneDimObj.Length / (len * wid);
            T[,,] threeDimObj = new T[len, wid, height];

            for (int i = 0; i < oneDimObj.Length; i++)
            {
                threeDimObj[i / (wid * height), (i / height) % wid, i % height] = oneDimObj[i];
            }
            return threeDimObj;
        }

        private List<VadOutputEntity> Infer(List<VadInputEntity> vadInputEntitys)
        {
            List<VadOutputEntity> vadOutputEntities = new List<VadOutputEntity>();
            foreach (VadInputEntity vadInputEntity in vadInputEntitys)
            {
                int batchSize = 1;//_batchSize                
                var inputMeta = _onnxSession.InputMetadata;
                var container = new List<NamedOnnxValue>();
                int[] dim = new int[] { batchSize, vadInputEntity.Speech.Length / 400 / batchSize, 400 };
                var tensor = new DenseTensor<float>(vadInputEntity.Speech, dim, false);
                container.Add(NamedOnnxValue.CreateFromTensor<float>("speech", tensor));

                int i = 0;
                foreach (var cache in vadInputEntity.InCaches)
                {
                    int[] cache_dim = new int[] { 1, 128, cache.Length / 128 / 1, 1 };
                    var cache_tensor = new DenseTensor<float>(cache, cache_dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>("in_cache" + i.ToString(), cache_tensor));
                    i++;
                }

                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _onnxSession.Run(container);
                var resultsArray = results.ToArray();
                VadOutputEntity vadOutputEntity = new VadOutputEntity();
                for (int j = 0; j < resultsArray.Length; j++)
                {
                    if (resultsArray[j].Name.Equals("logits"))
                    {
                        Tensor<float> tensors = resultsArray[0].AsTensor<float>();
                        var _scores = DimOneToThree<float>(tensors.ToArray(), 1, tensors.Dimensions[1]);
                        vadOutputEntity.Scores = _scores;
                    }
                    if (resultsArray[j].Name.StartsWith("out_cache"))
                    {
                        vadOutputEntity.OutCaches.Add(resultsArray[j].AsEnumerable<float>().ToArray());
                    }

                }
                vadOutputEntities.Add(vadOutputEntity);
            }

            return vadOutputEntities;
        }

        private float[] PadSequence(List<VadInputEntity> modelInputs)
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
                // ///////////////////////////////////////////////////
                var arr_neg_mean = _onnxSession.ModelMetadata.CustomMetadataMap["neg_mean"].ToString().Split(',').ToArray();
                double[] neg_mean = arr_neg_mean.Select(x => (double)Convert.ToDouble(x)).ToArray();
                var arr_inv_stddev = _onnxSession.ModelMetadata.CustomMetadataMap["inv_stddev"].ToString().Split(',').ToArray();
                double[] inv_stddev = arr_inv_stddev.Select(x => (double)Convert.ToDouble(x)).ToArray();

                int dim = neg_mean.Length;
                for (int j = 0; j < max_speech_length; j++)
                {
                    int k = new Random().Next(0, dim);
                    padspeech[j] = (float)((float)(0 + neg_mean[k]) * inv_stddev[k]);
                }
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
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    if (_onnxSession != null)
                    {
                        _onnxSession.Dispose();
                    }
                    if (_wavFrontend != null)
                    {
                        _wavFrontend.Dispose();
                    }
                    if (_encoderConfEntity != null)
                    {
                        _encoderConfEntity = null;
                    }
                    if (_vad_post_conf != null)
                    {
                        _vad_post_conf = null;
                    }
                }
                _disposed = true;
            }
        }

        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
        ~AliFsmnVad()
        {
            Dispose(_disposed);
        }
    }
}
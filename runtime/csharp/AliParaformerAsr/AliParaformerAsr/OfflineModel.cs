// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
using Microsoft.ML.OnnxRuntime;

namespace AliParaformerAsr
{
    public enum OnnxRumtimeTypes
    {
        CPU = 0,

        DML = 1,

        CUDA = 2,
    }
    public class OfflineModel
    {
        private InferenceSession _modelSession;
        private int _blank_id = 0;
        private int sos_eos_id = 1;
        private int _unk_id = 2;
        private int _featureDim = 80;
        private int _sampleRate = 16000;

        public OfflineModel(string modelFilePath, int threadsNum = 2, OnnxRumtimeTypes rumtimeType = OnnxRumtimeTypes.CPU, int deviceId = 0)
        {
            _modelSession = initModel(modelFilePath, threadsNum, rumtimeType, deviceId);
        }
        public int Blank_id { get => _blank_id; set => _blank_id = value; }
        public int Sos_eos_id { get => sos_eos_id; set => sos_eos_id = value; }
        public int Unk_id { get => _unk_id; set => _unk_id = value; }
        public int FeatureDim { get => _featureDim; set => _featureDim = value; }
        public InferenceSession ModelSession { get => _modelSession; set => _modelSession = value; }
        public int SampleRate { get => _sampleRate; set => _sampleRate = value; }

        public InferenceSession initModel(string modelFilePath, int threadsNum = 2, OnnxRumtimeTypes rumtimeType = OnnxRumtimeTypes.CPU, int deviceId = 0)
        {
            var options = new SessionOptions();
            switch (rumtimeType)
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
            options.InterOpNumThreads = threadsNum;
            InferenceSession onnxSession = new InferenceSession(modelFilePath, options);
            return onnxSession;
        }
        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                if (_modelSession != null)
                {
                    _modelSession.Dispose();
                }
            }
        }

        internal void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}

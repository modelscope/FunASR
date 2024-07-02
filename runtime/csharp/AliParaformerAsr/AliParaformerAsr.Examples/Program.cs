using AliParaformerAsr;
using CommandLine;
using NAudio.Wave;

internal static class Program
{

    public class ProgramParams
    {
        [Option('i', "input", Required = true, HelpText = "Input wav file/folder path.")]
        public string WavFilePath { get; set; }

        [Option('m', "model", Default = "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx", HelpText = "Model path.")]
        public string Model { get; set; }
    }

    [STAThread]
    private static void Main(string[] args)
    {
        var argParams = Parser.Default.ParseArguments<ProgramParams>(args).Value;

        string modelPath = argParams.Model;
        if (!Directory.Exists(argParams.Model))
        {
            modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, modelPath);
            if (!Directory.Exists(modelPath))
            {
                throw new DirectoryNotFoundException($"Model not found: {argParams.Model}");
            }
        }

        string modelFilePath = Path.Combine(modelPath, "model_quant.onnx");
        string configFilePath = Path.Combine(modelPath, "asr.yaml");
        string mvnFilePath = Path.Combine(modelPath, "am.mvn");
        string tokensFilePath = Path.Combine(modelPath, "tokens.json");

        
        var offlineRecognizer = new OfflineRecognizer(modelFilePath, configFilePath, mvnFilePath, tokensFilePath, OnnxRumtimeTypes.CPU);

        List<float[]> samples = new List<float[]>();
        TimeSpan total_duration = new TimeSpan(0L);

        if (File.Exists(argParams.WavFilePath))
        {
            (var sample, var duration) = LoadWavFile(argParams.WavFilePath);

            samples.Add(sample);
            total_duration += duration;
        }
        else if (Directory.Exists(argParams.WavFilePath)) 
        {
            var findWavCount = 0;

            foreach (var wavFilePath in Directory.EnumerateFiles(argParams.WavFilePath, "*.wav"))
            {
                (var sample, var duration) = LoadWavFile(wavFilePath);

                samples.Add(sample);
                total_duration += duration;
                findWavCount++;
            }

            Console.WriteLine($"Total WAV files found: {findWavCount} duration：{total_duration}");
        }
        else
        {
            throw new Exception($"Invalid wav input path. {argParams.WavFilePath}");
        }

        var start_time = DateTime.Now;

        int batchSize = 1; // 输入参数支持批处理，但是实际效果提升有限，感觉还是负优化，等GPU版本优化后再试
        for (int i = 0; i < samples.Count; i += batchSize)
        {
            List<float[]> temp_samples = samples.Skip(i).Take(batchSize).ToList();

            List<string> results = offlineRecognizer.GetResults(temp_samples);

            foreach (string result in results)
            {
                Console.WriteLine(result);
                Console.WriteLine("");
            }
        }


        var end_time = DateTime.Now;

        double elapsed_milliseconds = (end_time - start_time).TotalMilliseconds;
        double rtf = elapsed_milliseconds / total_duration.TotalMilliseconds;

        Console.WriteLine("elapsed_milliseconds:{0}", elapsed_milliseconds.ToString());
        Console.WriteLine("total_duration:{0}", total_duration.TotalMilliseconds.ToString());

        // 实时因子是处理时间与音频时长的比值。
        // 例如，如果一个 10 秒的音频片段需要 5 秒来处理，那么实时因子就是 0.5。
        // 如果处理时间和音频时长相等，那么实时因子就是 1，这意味着系统以实时速度进行处理。 
        // 数值越小，表示处理速度越快。
        // from chatgpt 解释
        Console.WriteLine("Real-Time Factor :{0}", rtf.ToString());
        Console.WriteLine("end!");
    }

    private static (float[] sample, TimeSpan duration) LoadWavFile(string wavFilePath)
    {
        AudioFileReader _audioFileReader = new AudioFileReader(wavFilePath);
        byte[] datas = new byte[_audioFileReader.Length];
        _audioFileReader.Read(datas, 0, datas.Length);
        var duration = _audioFileReader.TotalTime;
        float[] wavdata = new float[datas.Length / 4];
        Buffer.BlockCopy(datas, 0, wavdata, 0, datas.Length);
        var sample = wavdata.Select((float x) => x * 32768f).ToArray();

        return (sample, duration);
    }
}

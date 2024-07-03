using AliFsmnVadSharp;
using AliFsmnVadSharp.Model;
using CommandLine;
using NAudio.Wave;

internal static class Program
{
    public class ProgramParams
    {
        [Option('i', "input", Required = true, HelpText = "Input wav file/folder path.")]
        public string WavFilePath { get; set; }

        [Option('m', "model", Default = "speech_fsmn_vad_zh-cn-16k-common-onnx", HelpText = "Model path.")]
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
        string configFilePath = Path.Combine(modelPath, "config.yaml");
        string mvnFilePath = Path.Combine(modelPath, "am.mvn");

        int batchSize = 1;
        AliFsmnVad aliFsmnVad = new AliFsmnVad(modelFilePath, configFilePath, mvnFilePath, batchSize);

        List<string> wavFiles = new List<string>();

        if (File.Exists(argParams.WavFilePath))
        {
            wavFiles.Add(argParams.WavFilePath);
        }
        else if (Directory.Exists(argParams.WavFilePath))
        {
            foreach (var wavFilePath in Directory.GetFiles(argParams.WavFilePath, "*.wav"))
            {
                wavFiles.Add(wavFilePath);
            }
        }
        else
        {
            throw new Exception($"Invalid wav input path. {argParams.WavFilePath}");
        }

        var start_time = DateTime.Now;

        TimeSpan total_duration = new TimeSpan(0L);
        for (int i = 0; i < wavFiles.Count; i += batchSize)
        {
            List<float[]> samples = new List<float[]>();
            
            foreach(var wavFile in wavFiles.Skip(i).Take(batchSize))
            {
                (var sample, var duration) = LoadWavFile(wavFile);
                samples.Add(sample);
                total_duration += duration;
            }

            SegmentEntity[] segments_duration = aliFsmnVad.GetSegments(samples);
            Console.WriteLine("vad infer result:");
            foreach (SegmentEntity segment in segments_duration)
            {
                Console.Write("[");
                foreach (var x in segment.Segment)
                {
                    Console.Write("[" + string.Join(",", x.ToArray()) + "]");
                }
                Console.Write("]\r\n");
            }
        }

        var end_time = DateTime.Now;

		double elapsed_milliseconds = (end_time - start_time).TotalMilliseconds;

		double rtf = elapsed_milliseconds / total_duration.TotalMilliseconds;
		Console.WriteLine("elapsed_milliseconds:{0}", elapsed_milliseconds.ToString());
		Console.WriteLine("total_duration:{0}", total_duration.TotalMilliseconds.ToString());
		Console.WriteLine("rtf:{1}", "0".ToString(), rtf.ToString());
		Console.WriteLine("------------------------");
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
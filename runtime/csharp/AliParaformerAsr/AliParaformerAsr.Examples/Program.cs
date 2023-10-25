using AliParaformerAsr;
using NAudio.Wave;
internal static class Program
{
	[STAThread]
	private static void Main()
	{
        string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
        string modelName = "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx";
        string modelFilePath = applicationBase + "./"+ modelName + "/model_quant.onnx";
        string configFilePath = applicationBase + "./" + modelName + "/asr.yaml";
        string mvnFilePath = applicationBase + "./" + modelName + "/am.mvn";
        string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
        AliParaformerAsr.OfflineRecognizer offlineRecognizer = new OfflineRecognizer(modelFilePath, configFilePath, mvnFilePath, tokensFilePath);
        List<float[]>? samples = null;
        TimeSpan total_duration = new TimeSpan(0L);
        if (samples == null)
        {
            samples = new List<float[]>();
            for (int i = 0; i < 5; i++)
            {
                string wavFilePath = string.Format(applicationBase + "./" + modelName + "/example/{0}.wav", i.ToString());
                if (!File.Exists(wavFilePath))
                {
                    break;
                }
                AudioFileReader _audioFileReader = new AudioFileReader(wavFilePath);
                byte[] datas = new byte[_audioFileReader.Length];
                _audioFileReader.Read(datas, 0, datas.Length);
                TimeSpan duration = _audioFileReader.TotalTime;
                float[] wavdata = new float[datas.Length / 4];
                Buffer.BlockCopy(datas, 0, wavdata, 0, datas.Length);
                float[] sample = wavdata.Select((float x) => x * 32768f).ToArray();
                samples.Add(sample);
                total_duration += duration;
            }
        }
        TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
        //1.Non batch method
        foreach (var sample in samples)
        {
            List<float[]> temp_samples = new List<float[]>();
            temp_samples.Add(sample);
            List<string> results = offlineRecognizer.GetResults(temp_samples);
            foreach (string result in results)
            {
                Console.WriteLine(result);
                Console.WriteLine("");
            }
        }
        //2.batch method
        //List<string> results_batch = offlineRecognizer.GetResults(samples);
        //foreach (string result in results_batch)
        //{
        //    Console.WriteLine(result);
        //    Console.WriteLine("");
        //}
        TimeSpan end_time = new TimeSpan(DateTime.Now.Ticks);
        double elapsed_milliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
        double rtf = elapsed_milliseconds / total_duration.TotalMilliseconds;
        Console.WriteLine("elapsed_milliseconds:{0}", elapsed_milliseconds.ToString());
        Console.WriteLine("total_duration:{0}", total_duration.TotalMilliseconds.ToString());
        Console.WriteLine("rtf:{1}", "0".ToString(), rtf.ToString());
        Console.WriteLine("end!");
    }
}
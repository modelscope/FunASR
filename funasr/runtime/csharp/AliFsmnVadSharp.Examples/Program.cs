using AliFsmnVadSharp;
using AliFsmnVadSharp.Model;
using NAudio.Wave;

internal static class Program
{
	[STAThread]
	private static void Main()
	{
		string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
		string modelFilePath = applicationBase + "./speech_fsmn_vad_zh-cn-16k-common-pytorch/model.onnx";
		string configFilePath = applicationBase + "./speech_fsmn_vad_zh-cn-16k-common-pytorch/vad.yaml";
		string mvnFilePath = applicationBase + "./speech_fsmn_vad_zh-cn-16k-common-pytorch/vad.mvn";
		int batchSize = 2;
		TimeSpan start_time0 = new TimeSpan(DateTime.Now.Ticks);
		AliFsmnVad aliFsmnVad = new AliFsmnVad(modelFilePath, configFilePath, mvnFilePath, batchSize);
		TimeSpan end_time0 = new TimeSpan(DateTime.Now.Ticks);
		double elapsed_milliseconds0 = end_time0.TotalMilliseconds - start_time0.TotalMilliseconds;
		Console.WriteLine("load model and init config elapsed_milliseconds:{0}", elapsed_milliseconds0.ToString());
		List<float[]> samples = new List<float[]>();
		TimeSpan total_duration = new TimeSpan(0L);
		for (int i = 0; i < 2; i++)
		{
			string wavFilePath = string.Format(applicationBase + "./speech_fsmn_vad_zh-cn-16k-common-pytorch/example/{0}.wav", i.ToString());//vad_example
			if (!File.Exists(wavFilePath))
			{
				continue;
			}
			AudioFileReader _audioFileReader = new AudioFileReader(wavFilePath);
			byte[] datas = new byte[_audioFileReader.Length];
			_audioFileReader.Read(datas, 0, datas.Length);
			TimeSpan duration = _audioFileReader.TotalTime;
			float[] wavdata = new float[datas.Length / 4];
			Buffer.BlockCopy(datas, 0, wavdata, 0, datas.Length);
			float[] sample = wavdata.Select((float x) => x * 32768f).ToArray();
			samples.Add(wavdata);
			total_duration += duration;			
		}
		TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
		//SegmentEntity[] segments_duration = aliFsmnVad.GetSegments(samples);
		SegmentEntity[] segments_duration = aliFsmnVad.GetSegmentsByStep(samples);
		TimeSpan end_time = new TimeSpan(DateTime.Now.Ticks);
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

		double elapsed_milliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
		double rtf = elapsed_milliseconds / total_duration.TotalMilliseconds;
		Console.WriteLine("elapsed_milliseconds:{0}", elapsed_milliseconds.ToString());
		Console.WriteLine("total_duration:{0}", total_duration.TotalMilliseconds.ToString());
		Console.WriteLine("rtf:{1}", "0".ToString(), rtf.ToString());
		Console.WriteLine("------------------------");
	}
}
using AliCTTransformerPunc;

internal static class Program
{
	[STAThread]
	private static void Main()
	{
        string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
        string modelName = "punc_ct-transformer_zh-cn-common-vocab272727-onnx";
        string modelFilePath = applicationBase + "./"+ modelName + "/model.onnx";
        string configFilePath = applicationBase + "./"+modelName+"/punc.yaml";
        string tokensFilePath = applicationBase + "./"+modelName+"/tokens.txt";
        TimeSpan start_time1 = new TimeSpan(DateTime.Now.Ticks);
        AliCTTransformerPunc.CTTransformer ctTransformer = new CTTransformer(modelFilePath, configFilePath, tokensFilePath);
        TimeSpan end_time1 = new TimeSpan(DateTime.Now.Ticks);
        double elapsed_milliseconds1 = end_time1.TotalMilliseconds - start_time1.TotalMilliseconds;
        Console.WriteLine("load_model_elapsed_milliseconds:{0}", elapsed_milliseconds1.ToString());
        TimeSpan total_duration = new TimeSpan(0L);
        string text = "跨境河流是养育沿岸人民的生命之源长期以来为帮助下游地区防灾减灾中方技术人员在上游地区极为恶劣的自然条件下克服巨大困难甚至冒着生命危险向印方提供汛期水文资料处理紧急事件中方重视印方在跨境河流问题上的关切愿意进一步完善双方联合工作机制凡是中方能做的我们都会去做而且会做得更好我请印度朋友们放心中国在上游的任何开发利用都会经过科学规划和论证兼顾上下游的利益";
        TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
        string result = ctTransformer.GetResults(text);
        Console.WriteLine(result);
        Console.WriteLine("");
        TimeSpan end_time = new TimeSpan(DateTime.Now.Ticks);
        double elapsed_milliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
        Console.WriteLine("elapsed_milliseconds:{0}", elapsed_milliseconds.ToString());
        Console.WriteLine("end!");
    }
}
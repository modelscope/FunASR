using Android.App;
using Android.Runtime;

namespace MauiApp1;

[Application]

//[assembly: UsesPermission(Android.Manifest.Permission.ReadExternalStorage, MaxSDKVersion = 32)]
[assembly: UsesPermission(Android.Manifest.Permission.ReadMediaAudio)]
[assembly: UsesPermission(Android.Manifest.Permission.ReadMediaImages)]
[assembly: UsesPermission(Android.Manifest.Permission.ReadMediaVideo)]
[assembly: UsesPermission(Android.Manifest.Permission.ManageExternalStorage)]
// Needed for Picking photo/video
[assembly: UsesPermission(Android.Manifest.Permission.ReadExternalStorage)]

// Needed for Taking photo/video
[assembly: UsesPermission(Android.Manifest.Permission.WriteExternalStorage)]
[assembly: UsesPermission(Android.Manifest.Permission.Camera)]
[assembly: UsesPermission(Android.Manifest.Permission.RecordAudio)]
[assembly: UsesPermission(Android.Manifest.Permission.CaptureVideoOutput)]

// Add these properties if you would like to filter out devices that do not have cameras, or set to false to make them optional
[assembly: UsesFeature("android.hardware.camera", Required = true)]
[assembly: UsesFeature("android.hardware.camera.autofocus", Required = true)]
[assembly: UsesFeature("android.hardware.recordaudio", Required = true)]
[assembly: UsesFeature("android.hardware.recordaudio.autofocus", Required = true)]
[assembly: UsesFeature("android.hardware.capturevideooutput", Required = true)]
[assembly: UsesFeature("android.hardware.capturevideooutput.autofocus", Required = true)]


public class MainApplication : MauiApplication
{
	public MainApplication(IntPtr handle, JniHandleOwnership ownership)
		: base(handle, ownership)
	{
	}

	protected override MauiApp CreateMauiApp() => MauiProgram.CreateMauiApp();
}

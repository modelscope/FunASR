// AliFsmnVadSharp, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null
// AliFsmnVadSharp.Model.E2EVadSpeechBufWithDoaEntity
internal class E2EVadSpeechBufWithDoaEntity
{
	private int _start_ms = 0;

	private int _end_ms = 0;

	private byte[]? _buffer;

	private bool _contain_seg_start_point = false;

	private bool _contain_seg_end_point = false;

	private int _doa = 0;

	public int start_ms
	{
		get
		{
			return _start_ms;
		}
		set
		{
			_start_ms = value;
		}
	}

	public int end_ms
	{
		get
		{
			return _end_ms;
		}
		set
		{
			_end_ms = value;
		}
	}

	public byte[]? buffer
	{
		get
		{
			return _buffer;
		}
		set
		{
			_buffer = value;
		}
	}

	public bool contain_seg_start_point
	{
		get
		{
			return _contain_seg_start_point;
		}
		set
		{
			_contain_seg_start_point = value;
		}
	}

	public bool contain_seg_end_point
	{
		get
		{
			return _contain_seg_end_point;
		}
		set
		{
			_contain_seg_end_point = value;
		}
	}

	public int doa
	{
		get
		{
			return _doa;
		}
		set
		{
			_doa = value;
		}
	}

	public void Reset()
	{
		_start_ms = 0;
		_end_ms = 0;
		_buffer = new byte[0];
		_contain_seg_start_point = false;
		_contain_seg_end_point = false;
		_doa = 0;
	}
}

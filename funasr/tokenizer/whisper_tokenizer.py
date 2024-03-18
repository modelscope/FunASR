
try:
	from whisper.tokenizer import get_tokenizer
except:
	print("Notice: If you want to use whisper, please `pip install -U openai-whisper`")

from funasr.register import tables

@tables.register("tokenizer_classes", "WhisperTokenizer")
def WhisperTokenizer(**kwargs):

	language = kwargs.get("language", None)
	task = kwargs.get("task", "transcribe")
	is_multilingual = kwargs.get("is_multilingual", True)
	num_languages = kwargs.get("num_languages", 99)
	tokenizer = get_tokenizer(
		multilingual=is_multilingual,
		num_languages=num_languages,
		language=language,
		task=task,
	)
	
	return tokenizer


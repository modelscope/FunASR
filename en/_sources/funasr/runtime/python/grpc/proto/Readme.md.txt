```	
service ASR { //grpc service
  rpc Recognize (stream Request) returns (stream Response) {} //Stub
}

message Request { //request data
  bytes audio_data = 1; //audio data in bytes.
  string user = 2; //user allowed.
  string language = 3; //language, zh-CN for now.
  bool speaking = 4; //flag for speaking. 
  bool isEnd = 5; //flag for end. set isEnd to true when you stop asr:
  //vad:is_speech then speaking=True & isEnd = False, audio data will be appended for the specfied user.
  //vad:silence then speaking=False & isEnd = False, clear audio buffer and do asr inference.
}

message Response { //response data.
  string sentence = 1; //json, includes flag for success and asr text .
  string user = 2; //same to request user.
  string language = 3; //same to request language.
  string action = 4; //server status:
  //terminate：asr stopped; 
  //speaking：user is speaking, audio data is appended; 
  //decoding: server is decoding; 
  //finish: get asr text, most used.
}

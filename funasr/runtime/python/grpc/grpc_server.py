from concurrent import futures
import grpc
import json
import paraformer_pb2
import paraformer_pb2_grpc
import time


from paraformer_pb2 import Response


from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_16k_pipline = pipeline(
   task=Tasks.auto_speech_recognition,
   model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1')

auth_user = ['zksz_futureTV_1','zksz_dangaomei_1','zksz_test_1','zksz_test_2','zksz_test_3','zksz_test_4','zksz_test_5','zksz_test_6','zksz_test_7']


class ASRServicer(paraformer_pb2_grpc.ASRServicer):
    def __init__(self):
        print("ASRServicer init")
        self.init_flag = 0
        self.client_buffers = {}
        self.client_transcription = {}

    def clear_states(self, user):
        self.clear_buffers(user)
        self.clear_transcriptions(user)

    def clear_buffers(self, user):
        if user in self.client_buffers:
            del self.client_buffers[user]

    def clear_transcriptions(self, user):
        if user in self.client_transcription:
            del self.client_transcription[user]

    def disconnect(self, user):
        self.clear_states(user)
        print("Disconnecting user: %s" % str(user))

    def Recognize(self, request_iterator, context):
        
            
        for req in request_iterator:
            if req.user not in auth_user:
                result = {}
                result["success"] = False
                result["detail"] = "Not Authorized user: %s " % req.user
                result["text"] = ""
                yield Response(sentence=json.dumps(result), user=req.user, action="terminate", language=req.language)
            if req.isEnd: #end grpc
                print("asr end")
                self.disconnect(req.user)
                result = {}
                result["success"] = True
                result["detail"] = "asr end"
                result["text"] = ""
                yield Response(sentence=json.dumps(result), user=req.user, action="terminate",language=req.language)
            elif req.speaking: #continue speaking
                if req.audio_data is not None and len(req.audio_data) > 0:
                    if req.user in self.client_buffers:
                        self.client_buffers[req.user] += req.audio_data #append audio
                    else:
                        self.client_buffers[req.user] = req.audio_data
                result = {}
                result["success"] = True
                result["detail"] = "speaking"
                result["text"] = ""
                yield Response(sentence=json.dumps(result), user=req.user, action="speaking", language=req.language)
            elif not req.speaking: #silence
                if req.user not in self.client_buffers:
                    result = {}
                    result["success"] = True
                    result["detail"] = "waiting_for_voice"
                    result["text"] = ""
                    yield Response(sentence=json.dumps(result), user=req.user, action="waiting", language=req.language)
                else:
                    begin_time = int(round(time.time() * 1000))
                    tmp_data = self.client_buffers[req.user] #TODO make a test, about local variable in class parralle circumstance.
                    self.clear_states(req.user)
                    result = {}
                    result["success"] = True
                    result["detail"] = "decoding data: %d bytes" % len(tmp_data)
                    result["text"] = ""
                    yield Response(sentence=json.dumps(result), user=req.user, action="decoding", language=req.language)
                    if len(tmp_data) < 800: #min input_len for asr model 
                        end_time = int(round(time.time() * 1000))
                        delay_str = str(end_time - begin_time)
                        result = {}
                        result["success"] = True
                        result["detail"] = "finish_sentence_data_is_not_long_enough"
                        result["server_delay_ms"] = delay_str
                        result["text"] = ""
                        print ("user: %s , delay(ms): %s, error: %s " % (req.user, delay_str, "data_is_not_long_enough"))
                        yield Response(sentence=json.dumps(result), user=req.user, action="finish", language=req.language)
                    else:                           
                        asr_result = inference_16k_pipline(audio_in=tmp_data, audio_fs = 16000)
                        if "text" in asr_result:
                            asr_result = asr_result['text']
                        else:
                            asr_result = ""
                        end_time = int(round(time.time() * 1000))
                        delay_str = str(end_time - begin_time)
                        print ("user: %s , delay(ms): %s, text: %s " % (req.user, delay_str, asr_result))
                        result = {}
                        result["success"] = True
                        result["detail"] = "finish_sentence"
                        result["server_delay_ms"] = delay_str
                        result["text"] = asr_result
                        yield Response(sentence=json.dumps(result), user=req.user, action="finish", language=req.language)
            else:
                result = {}
                result["success"] = False 
                result["detail"] = "error, no condition matched! Unknown reason."
                result["text"] = ""
                self.disconnect(req.user)
                yield Response(sentence=json.dumps(result), user=req.user, action="terminate", language=req.language)
                


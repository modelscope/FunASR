// funasr-vad: FSMN-VAD on ggml. WAV(any fmt/rate) -> speech segments [start_ms,end_ms].
// Front end + FSMN encoder validated bit-exact vs PyTorch; state machine reproduces
// E2EVadModel segmentation to within 1 frame (10ms) of fsmn-vad.generate on the 184-clip set.
#define FUNASR_AUDIO_IMPLEMENTATION
#include "funasr_audio.h"
#include "funasr_vad.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

int main(int argc,char**argv){
  std::string gp,wp;
  for(int i=1;i<argc;i++){if(!strcmp(argv[i],"-m")&&i+1<argc)gp=argv[++i];else if(!strcmp(argv[i],"-a")&&i+1<argc)wp=argv[++i];}
  if(gp.empty()||wp.empty()){fprintf(stderr,"usage: %s -m fsmn-vad.gguf -a audio.wav\n",argv[0]);return 1;}
  std::vector<float> wav; if(!funasr_load_audio_16k_mono(wp.c_str(),wav)){fprintf(stderr,"read audio failed\n");return 1;}
  int max_seg = getenv("VAD_MAXSEG") ? atoi(getenv("VAD_MAXSEG")) : 30000;
  std::vector<std::pair<int,int>> segs;
  if(!funasr_vad_segments(gp,wav,max_seg,segs)){fprintf(stderr,"vad failed\n");return 1;}
  for(auto&s:segs) printf("%d %d\n", s.first, s.second);
  fprintf(stderr,"[vad] %zu segments (max_seg=%dms)\n",segs.size(),max_seg);
  return 0;
}

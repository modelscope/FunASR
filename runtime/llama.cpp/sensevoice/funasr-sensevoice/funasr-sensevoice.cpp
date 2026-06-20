// funasr-sensevoice: SenseVoiceSmall (SAN-M encoder + CTC) on ggml.
//   fbank.bin (T x 560) -> CMVN -> prepend 4 query tokens -> SAN-M encoder ->
//   CTC head -> greedy CTC decode -> token ids (stdout).
// The encoder is the same SAN-M arch as Fun-ASR-Nano (shared forward).
// Detokenize the printed ids with the SentencePiece bpe model (Python side for now).

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

static const float LN_EPS = 1e-5f;

// ---- audio loader: any wav/mp3/flac, any rate/channels -> 16k mono (miniaudio) ----
#define FUNASR_AUDIO_IMPLEMENTATION
#include "funasr_audio.h"
#include "funasr_vad.h"     // built-in FSMN-VAD front end (--vad segmentation)
#include <utility>
static const int FS=16000,WINLEN=400,SHIFT=160,NFFT=512,NMEL=80,LFR_M=7,LFR_N=6;
static const float PREEMPH=0.97f,LOWF=20.0f,HIGHF=8000.0f;
static inline float melf(float f){return 1127.0f*logf(1.0f+f/700.0f);}
static void fftc(std::vector<float>&re,std::vector<float>&im,int n){
  for(int i=1,j=0;i<n;i++){int b=n>>1;for(;j&b;b>>=1)j^=b;j^=b;if(i<j){std::swap(re[i],re[j]);std::swap(im[i],im[j]);}}
  for(int len=2;len<=n;len<<=1){double a=-2.0*M_PI/len;float wr=cosf(a),wi=sinf(a);
    for(int i=0;i<n;i+=len){float cr=1,ci=0;for(int k=0;k<len/2;k++){float ur=re[i+k],ui=im[i+k];
      float vr=re[i+k+len/2]*cr-im[i+k+len/2]*ci,vi=re[i+k+len/2]*ci+im[i+k+len/2]*cr;
      re[i+k]=ur+vr;im[i+k]=ui+vi;re[i+k+len/2]=ur-vr;im[i+k+len/2]=ui-vi;float nc=cr*wr-ci*wi;ci=cr*wi+ci*wr;cr=nc;}}}
}
static std::vector<float> compute_fbank(std::vector<float> wav,int&T_out){
  for(auto&v:wav)v*=32768.0f; std::vector<float> win(WINLEN);
  for(int i=0;i<WINLEN;i++)win[i]=0.54f-0.46f*cosf(2.0f*M_PI*i/(WINLEN-1));
  const int NBIN=NFFT/2+1; float bw=(float)FS/NFFT,ml=melf(LOWF),mh=melf(HIGHF),dm=(mh-ml)/(NMEL+1);
  std::vector<std::vector<float>> fb(NMEL,std::vector<float>(NBIN,0.0f));
  for(int m=0;m<NMEL;m++){float L=ml+m*dm,C=ml+(m+1)*dm,R=ml+(m+2)*dm;
    for(int k=0;k<NBIN;k++){float mf=melf(bw*k); if(mf>L&&mf<R)fb[m][k]=mf<=C?(mf-L)/(C-L):(R-mf)/(R-C);}}
  int N=wav.size(),T=(N-WINLEN)/SHIFT+1; std::vector<std::vector<float>> feat(T,std::vector<float>(NMEL));
  std::vector<float> re(NFFT),im(NFFT),fr(WINLEN); const float fl=1.1920929e-07f;
  for(int t=0;t<T;t++){const float*s=wav.data()+t*SHIFT; double mn=0; for(int i=0;i<WINLEN;i++)mn+=s[i]; mn/=WINLEN;
    for(int i=0;i<WINLEN;i++)fr[i]=s[i]-(float)mn; for(int i=WINLEN-1;i>0;i--)fr[i]-=PREEMPH*fr[i-1]; fr[0]-=PREEMPH*fr[0];
    for(int i=0;i<NFFT;i++){re[i]=i<WINLEN?fr[i]*win[i]:0.0f;im[i]=0.0f;} fftc(re,im,NFFT);
    for(int m=0;m<NMEL;m++){float e=0;for(int k=0;k<NBIN;k++)if(fb[m][k]>0)e+=fb[m][k]*(re[k]*re[k]+im[k]*im[k]); feat[t][m]=logf(e>fl?e:fl);}}
  const int pad=(LFR_M-1)/2; int Tl=(T+LFR_N-1)/LFR_N; std::vector<std::vector<float>> pd; pd.reserve(T+pad+LFR_M);
  for(int i=0;i<pad;i++)pd.push_back(feat[0]); for(int t=0;t<T;t++)pd.push_back(feat[t]);
  while((int)pd.size()<(Tl-1)*LFR_N+LFR_M)pd.push_back(feat[T-1]);
  int D=LFR_M*NMEL; std::vector<float> out((size_t)Tl*D);
  for(int i=0;i<Tl;i++)for(int j=0;j<LFR_M;j++)memcpy(&out[(size_t)i*D+j*NMEL],pd[i*LFR_N+j].data(),NMEL*sizeof(float));
  T_out=Tl; return out;
}

struct cfg { int d_model=512,n_head=4,num_blocks=50,tp_blocks=20,kernel=11,vocab=25055,blank=0; };
struct model { cfg c; ggml_context*ctx_w=nullptr; std::map<std::string,ggml_tensor*> t;
  ggml_tensor* g(const std::string&n){auto it=t.find(n);if(it==t.end()){fprintf(stderr,"missing %s\n",n.c_str());exit(1);}return it->second;} };

static ggml_tensor* lin(ggml_context*c,ggml_tensor*w,ggml_tensor*b,ggml_tensor*x){auto y=ggml_mul_mat(c,w,x);return b?ggml_add(c,y,b):y;}
static ggml_tensor* lnorm(ggml_context*c,ggml_tensor*x,ggml_tensor*g,ggml_tensor*b){return ggml_add(c,ggml_mul(c,ggml_norm(c,x,LN_EPS),g),b);}
static ggml_tensor* sanm_attn(ggml_context*c,model&m,const std::string&p,ggml_tensor*x,int T){
  const int D=m.c.d_model,H=m.c.n_head,dk=D/H,K=m.c.kernel;
  ggml_tensor*qkv=lin(c,m.g(p+"linear_q_k_v.weight"),m.g(p+"linear_q_k_v.bias"),x); size_t nb1=qkv->nb[1];
  ggml_tensor*q=ggml_cont(c,ggml_view_2d(c,qkv,D,T,nb1,0));
  ggml_tensor*k=ggml_cont(c,ggml_view_2d(c,qkv,D,T,nb1,(size_t)D*sizeof(float)));
  ggml_tensor*v=ggml_cont(c,ggml_view_2d(c,qkv,D,T,nb1,(size_t)2*D*sizeof(float)));
  const int pad=(K-1)/2; ggml_tensor*fk=m.g(p+"fsmn_block.weight");
  ggml_tensor*vp=ggml_pad_ext(c,v,0,0,pad,pad,0,0,0,0); ggml_tensor*fsmn=v;
  for(int j=0;j<K;j++){auto sl=ggml_view_2d(c,vp,D,T,vp->nb[1],(size_t)j*vp->nb[1]);
    auto wj=ggml_view_1d(c,fk,D,(size_t)j*fk->nb[1]); fsmn=ggml_add(c,fsmn,ggml_mul(c,ggml_cont(c,sl),wj));}
  q=ggml_permute(c,ggml_reshape_3d(c,q,dk,H,T),0,2,1,3); k=ggml_permute(c,ggml_reshape_3d(c,k,dk,H,T),0,2,1,3);
  ggml_tensor*vh=ggml_cont(c,ggml_permute(c,ggml_reshape_3d(c,v,dk,H,T),1,2,0,3));
  ggml_tensor*kq=ggml_soft_max(c,ggml_scale(c,ggml_mul_mat(c,k,q),1.0f/sqrtf((float)dk)));
  ggml_tensor*o=ggml_cont_2d(c,ggml_permute(c,ggml_mul_mat(c,vh,kq),0,2,1,3),D,T);
  return ggml_add(c,lin(c,m.g(p+"linear_out.weight"),m.g(p+"linear_out.bias"),o),fsmn);
}
static ggml_tensor* sanm_layer(ggml_context*c,model&m,const std::string&p,ggml_tensor*x,int T,bool res){
  auto r=x; auto h=lnorm(c,x,m.g(p+"norm1.weight"),m.g(p+"norm1.bias"));
  auto sa=sanm_attn(c,m,p+"self_attn.",h,T); x=res?ggml_add(c,r,sa):sa; r=x;
  h=lnorm(c,x,m.g(p+"norm2.weight"),m.g(p+"norm2.bias"));
  h=lin(c,m.g(p+"feed_forward.w_1.weight"),m.g(p+"feed_forward.w_1.bias"),h); h=ggml_relu(c,h);
  h=lin(c,m.g(p+"feed_forward.w_2.weight"),m.g(p+"feed_forward.w_2.bias"),h); return ggml_add(c,r,h);
}
static void add_posenc(std::vector<float>&x,int T,int depth){
  double inc=log(10000.0)/(depth/2.0-1.0);
  for(int t=0;t<T;t++){double pos=t+1;for(int i=0;i<depth/2;i++){double its=exp(i*-inc),st=pos*its;
    x[(size_t)t*depth+i]+=(float)sin(st);x[(size_t)t*depth+depth/2+i]+=(float)cos(st);}}
}

// SenseVoice detok: sentencepiece pieces (no byte-fallback in this vocab) -> join,
// "▁"(U+2581)->space; meta tokens <|lang|>/<|emo|>/<|event|>/<|itn|> dropped unless --keep-tags.
static std::string sv_trim(const std::string&s){size_t a=s.find_first_not_of(' ');if(a==std::string::npos)return "";size_t b=s.find_last_not_of(' ');return s.substr(a,b-a+1);}
static std::string detok_sv(const std::vector<int>&ids,const std::vector<std::string>&vocab,bool keep_tags){
  std::string s; for(int id:ids){ if(id<0||id>=(int)vocab.size())continue; const std::string&p=vocab[id];
    if(!keep_tags && p.size()>=2 && p[0]=='<' && p[1]=='|') continue;   // skip <|...|> meta
    s+=p; }
  const std::string lb="\xe2\x96\x81"; size_t pp; while((pp=s.find(lb))!=std::string::npos)s.replace(pp,3," ");
  return sv_trim(s);
}

int main(int argc,char**argv){
  std::string gguf_path,fbank_path,wav_path,vad_path; int vad_maxseg=30000; bool ids_mode=false,keep_tags=false;
  for(int i=1;i<argc;i++){ if(!strcmp(argv[i],"-m")&&i+1<argc)gguf_path=argv[++i];
    else if(!strcmp(argv[i],"-f")&&i+1<argc)fbank_path=argv[++i];
    else if(!strcmp(argv[i],"-a")&&i+1<argc)wav_path=argv[++i];
    else if(!strcmp(argv[i],"--vad")&&i+1<argc)vad_path=argv[++i];
    else if(!strcmp(argv[i],"--vad-maxseg")&&i+1<argc)vad_maxseg=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--ids"))ids_mode=true;
    else if(!strcmp(argv[i],"--keep-tags"))keep_tags=true;
    else {fprintf(stderr,"usage: %s -m sensevoice.gguf (-a audio.wav | -f fbank.bin) [--vad fsmn-vad.gguf [--vad-maxseg ms]] [--ids] [--keep-tags]\n",argv[0]);return 1;} }
  if(gguf_path.empty()||(fbank_path.empty()&&wav_path.empty())){fprintf(stderr,"missing args\n");return 1;}

  // load model
  model m; gguf_init_params gp={false,&m.ctx_w}; gguf_context*gg=gguf_init_from_file(gguf_path.c_str(),gp);
  if(!gg){fprintf(stderr,"load gguf failed\n");return 1;}
  auto rd=[&](const char*k,int d){int i=gguf_find_key(gg,k);return i<0?d:(int)gguf_get_val_u32(gg,i);};
  m.c.d_model=rd("sv.output_size",512); m.c.n_head=rd("sv.attention_heads",4);
  m.c.num_blocks=rd("sv.num_blocks",50); m.c.tp_blocks=rd("sv.tp_blocks",20);
  m.c.kernel=rd("sv.kernel_size",11); m.c.vocab=rd("sv.vocab_size",25055); m.c.blank=rd("sv.blank_id",0);
  int qi=gguf_find_key(gg,"sv.query_tokens"); int nq=qi<0?0:(int)gguf_get_arr_n(gg,qi);
  std::vector<int> qtok(nq); for(int i=0;i<nq;i++) qtok[i]=((const int32_t*)gguf_get_arr_data(gg,qi))[i];
  std::vector<std::string> vocab; {int ki=gguf_find_key(gg,"sv.vocab"); if(ki>=0){int nv=gguf_get_arr_n(gg,ki); vocab.resize(nv); for(int i=0;i<nv;i++)vocab[i]=gguf_get_arr_str(gg,ki,i);}}
  for(int i=0;i<gguf_get_n_tensors(gg);i++){const char*nm=gguf_get_tensor_name(gg,i);m.t[nm]=ggml_get_tensor(m.ctx_w,nm);}
  gguf_free(gg);
  const int F=560, D=m.c.d_model, V=m.c.vocab;
  bool emit_ids = ids_mode || vocab.empty();   // fall back to ids if the gguf has no vocab

  // NOTE: SenseVoiceSmall inference() feeds the RAW log-mel fbank to the encoder;
  // it does NOT apply am.mvn CMVN (that path is unused at inference). Applying it
  // makes the encoder predict <|nospeech|>. So no CMVN here.
  float*emb=(float*)m.g("embed.weight")->data;   // [16, 560] row-major
  // Run encoder+CTC on one fbank window [T,F]; prints greedy-CTC token IDs (no newline).
  auto run_seg=[&](const std::vector<float>& fb,int T){
    int N=nq+T; std::vector<float> inp((size_t)N*F);
    for(int i=0;i<nq;i++) memcpy(&inp[(size_t)i*F], &emb[(size_t)qtok[i]*F], F*sizeof(float));
    memcpy(&inp[(size_t)nq*F], fb.data(), (size_t)T*F*sizeof(float));
    float sc=sqrtf((float)D); for(auto&v:inp)v*=sc; add_posenc(inp,N,F);
    ggml_backend_t be=ggml_backend_cpu_init();
    ggml_init_params cp={(size_t)1024*1024*1024,nullptr,true}; ggml_context*c=ggml_init(cp);
    ggml_tensor*x=ggml_new_tensor_2d(c,GGML_TYPE_F32,F,N); ggml_set_input(x);
    ggml_tensor*h=sanm_layer(c,m,"encoder.encoders0.0.",x,N,false);
    for(int i=0;i<m.c.num_blocks-1;i++) h=sanm_layer(c,m,"encoder.encoders."+std::to_string(i)+".",h,N,true);
    h=lnorm(c,h,m.g("encoder.after_norm.weight"),m.g("encoder.after_norm.bias"));
    for(int i=0;i<m.c.tp_blocks;i++) h=sanm_layer(c,m,"encoder.tp_encoders."+std::to_string(i)+".",h,N,true);
    h=lnorm(c,h,m.g("encoder.tp_norm.weight"),m.g("encoder.tp_norm.bias"));
    ggml_tensor*logits=lin(c,m.g("ctc.ctc_lo.weight"),m.g("ctc.ctc_lo.bias"),h);  // [V, N]
    ggml_set_output(logits);
    ggml_cgraph*gf=ggml_new_graph_custom(c,32768,false); ggml_build_forward_expand(gf,logits);
    ggml_gallocr_t ga=ggml_gallocr_new(ggml_backend_cpu_buffer_type()); ggml_gallocr_alloc_graph(ga,gf);
    ggml_backend_tensor_set(x,inp.data(),0,ggml_nbytes(x)); ggml_backend_cpu_set_n_threads(be,8);
    if(ggml_backend_graph_compute(be,gf)!=GGML_STATUS_SUCCESS){fprintf(stderr,"compute failed\n");}
    std::vector<float> lg((size_t)V*N); ggml_backend_tensor_get(logits,lg.data(),0,ggml_nbytes(logits));
    std::vector<int> seg_ids; int prev=-1;   // greedy CTC: argmax per frame -> collapse -> drop blank
    for(int n=0;n<N;n++){ const float*col=&lg[(size_t)n*V]; int am=0; float best=col[0];
      for(int v=1;v<V;v++) if(col[v]>best){best=col[v];am=v;}
      if(am!=prev && am!=m.c.blank) seg_ids.push_back(am); prev=am; }
    if(emit_ids){ for(int id:seg_ids) printf("%d ",id); }
    else { std::string t=detok_sv(seg_ids,vocab,keep_tags); printf("%s",t.c_str()); }
    ggml_gallocr_free(ga); ggml_free(c); ggml_backend_free(be);
  };

  int64_t t0=ggml_time_us();
  if(!vad_path.empty()){
    std::vector<float> wav; if(!funasr_load_audio_16k_mono(wav_path.c_str(),wav)){fprintf(stderr,"read audio failed\n");return 1;}
    std::vector<std::pair<int,int>> segs;
    if(!funasr_vad_segments(vad_path,wav,vad_maxseg,segs)){fprintf(stderr,"vad failed\n");return 1;}
    for(auto&s:segs){ int off=(int)((int64_t)s.first*16000/1000), end=(int)((int64_t)s.second*16000/1000);
      if(end>(int)wav.size())end=wav.size(); if(end-off<WINLEN)continue;
      std::vector<float> seg(wav.begin()+off,wav.begin()+end); int t=0; auto fb=compute_fbank(seg,t); run_seg(fb,t); }
    fprintf(stderr,"[sensevoice] %zu vad segments\n",segs.size());
  } else {
    int32_t T=0,Fc=F; std::vector<float> fb;
    if(!wav_path.empty()){
      std::vector<float> wav; if(!funasr_load_audio_16k_mono(wav_path.c_str(),wav)){fprintf(stderr,"read audio failed\n");return 1;}
      int t=0; fb=compute_fbank(wav,t); T=t;
    } else {
      FILE*f=fopen(fbank_path.c_str(),"rb"); if(!f){fprintf(stderr,"open fbank\n");return 1;}
      if(fread(&T,4,1,f)!=1||fread(&Fc,4,1,f)!=1){fclose(f);return 1;}
      fb.resize((size_t)T*Fc); if((int)fread(fb.data(),4,fb.size(),f)!=(int)fb.size()){fclose(f);return 1;} fclose(f);
    }
    run_seg(fb,T);
  }
  printf("\n");
  fprintf(stderr,"[sensevoice] done %.2fs\n",(ggml_time_us()-t0)/1e6);
  if(m.ctx_w) ggml_free(m.ctx_w);
  return 0;
}

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

#include <cctype>
#include <cstdarg>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

static const float LN_EPS = 1e-5f;

static void trace_stage(const char *format, ...) {
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
  fputc('\n', stderr);
  fflush(stderr);
}

// ---- audio loader: any wav/mp3/flac, any rate/channels -> 16k mono (miniaudio) ----
#define FUNASR_AUDIO_IMPLEMENTATION
#include "funasr_audio.h"
#include "funasr_vad.h"     // built-in FSMN-VAD front end (--vad segmentation)
#include "funasr_srt.h"
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
struct model { cfg c; gguf_context*gguf=nullptr; ggml_context*ctx_meta=nullptr; ggml_context*ctx_w=nullptr;
  ggml_backend_buffer_t weights_buffer=nullptr; std::map<std::string,ggml_tensor*> t;
  ggml_tensor* g(const std::string&n){auto it=t.find(n);if(it==t.end()){fprintf(stderr,"missing %s\n",n.c_str());exit(1);}return it->second;} };

struct graph_backend {
  ggml_backend_t backend=nullptr;
  ggml_backend_buffer_type_t buffer_type=nullptr;
  bool is_cpu=true;
};

static std::string lower_copy(const char*s){
  std::string out=s?s:"";
  for(char&c:out)c=(char)std::tolower((unsigned char)c);
  return out;
}

static ggml_backend_dev_t find_gpu_backend_device(const std::string&backend_name){
  ggml_backend_load_all();
  ggml_backend_dev_t integrated_fallback=nullptr;
  for(size_t i=0;i<ggml_backend_dev_count();i++){
    ggml_backend_dev_t dev=ggml_backend_dev_get(i);
    enum ggml_backend_dev_type type=ggml_backend_dev_type(dev);
    if(type!=GGML_BACKEND_DEVICE_TYPE_GPU&&type!=GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
    std::string reg=lower_copy(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)));
    std::string dev_name=lower_copy(ggml_backend_dev_name(dev));
    std::string dev_desc=lower_copy(ggml_backend_dev_description(dev));
    if(reg.find(backend_name)!=std::string::npos||
       dev_name.find(backend_name)!=std::string::npos||
       dev_desc.find(backend_name)!=std::string::npos){
      if(type==GGML_BACKEND_DEVICE_TYPE_GPU) return dev;
      if(!integrated_fallback) integrated_fallback=dev;
    }
  }
  return integrated_fallback;
}

static graph_backend initialize_device_backend(const std::string&name,ggml_backend_dev_t dev){
  graph_backend out;
  const char*dev_name=ggml_backend_dev_name(dev);
  const char*dev_desc=ggml_backend_dev_description(dev);
  fprintf(stderr,"initializing %s backend on %s (%s)\n",name.c_str(),dev_name,dev_desc);
  fflush(stderr);
  out.backend=ggml_backend_dev_init(dev,nullptr);
  if(!out.backend){
    fprintf(stderr,"failed to initialize %s backend on %s\n",name.c_str(),dev_name);
    exit(1);
  }
  fprintf(stderr,"initialized %s backend on %s; resolving buffer type\n",name.c_str(),dev_name);
  fflush(stderr);
  out.buffer_type=ggml_backend_get_default_buffer_type(out.backend);
  if(!out.buffer_type){
    fprintf(stderr,"%s backend on %s has no default buffer type\n",name.c_str(),dev_name);
    exit(1);
  }
  fprintf(stderr,"%s backend ready on %s\n",name.c_str(),dev_name);
  fflush(stderr);
  out.is_cpu=false;
  return out;
}

static graph_backend make_graph_backend(const std::string&name){
  graph_backend out;
  if(name=="cpu"){
    out.backend=ggml_backend_cpu_init();
    if(!out.backend){
      fprintf(stderr,"failed to initialize cpu backend\n");
      exit(1);
    }
    out.buffer_type=ggml_backend_get_default_buffer_type(out.backend);
    out.is_cpu=true;
  } else if(name=="cuda"){
    ggml_backend_dev_t dev=find_gpu_backend_device("cuda");
    if(!dev){
      fprintf(stderr,"CUDA backend requested, but no GPU backend is available; build with -DGGML_CUDA=ON\n");
      exit(1);
    }
    return initialize_device_backend(name,dev);
  } else if(name=="vulkan"){
    ggml_backend_dev_t dev=find_gpu_backend_device("vulkan");
    if(!dev){
      fprintf(stderr,"Vulkan backend requested, but no Vulkan GPU backend is available; build with -DGGML_VULKAN=ON and install a Vulkan driver/ICD\n");
      exit(1);
    }
    return initialize_device_backend(name,dev);
  } else {
    fprintf(stderr,"unsupported backend '%s' (expected cpu|cuda|vulkan)\n",name.c_str());
    exit(1);
  }
  if(!out.backend||!out.buffer_type){
    fprintf(stderr,"failed to initialize %s backend\n",name.c_str());
    exit(1);
  }
  return out;
}

static void free_model(model&m){
  if(m.gguf){gguf_free(m.gguf);m.gguf=nullptr;}
  if(m.ctx_meta){ggml_free(m.ctx_meta);m.ctx_meta=nullptr;}
  if(m.weights_buffer){ggml_backend_buffer_free(m.weights_buffer);m.weights_buffer=nullptr;}
  if(m.ctx_w){ggml_free(m.ctx_w);m.ctx_w=nullptr;}
  m.t.clear();
}

static bool load_model_weights(const std::string&path,ggml_backend_buffer_type_t buffer_type,model&m){
  gguf_init_params gp={true,&m.ctx_meta};
  m.gguf=gguf_init_from_file(path.c_str(),gp);
  if(!m.gguf){fprintf(stderr,"load gguf failed\n");return false;}
  const int64_t n_tensors=gguf_get_n_tensors(m.gguf);
  ggml_init_params wp={(size_t)(n_tensors+1)*ggml_tensor_overhead(),nullptr,true};
  m.ctx_w=ggml_init(wp);
  if(!m.ctx_w){fprintf(stderr,"failed to initialize model tensor context\n");free_model(m);return false;}
  for(int64_t i=0;i<n_tensors;i++){
    const char*name=gguf_get_tensor_name(m.gguf,i);
    ggml_tensor*meta=ggml_get_tensor(m.ctx_meta,name);
    if(!meta){fprintf(stderr,"missing model tensor metadata: %s\n",name);free_model(m);return false;}
    ggml_tensor*weight=ggml_dup_tensor(m.ctx_w,meta);
    ggml_set_name(weight,name);
    m.t[name]=weight;
  }
  m.weights_buffer=ggml_backend_alloc_ctx_tensors_from_buft(m.ctx_w,buffer_type);
  if(!m.weights_buffer){fprintf(stderr,"failed to allocate model weights buffer\n");free_model(m);return false;}
  ggml_backend_buffer_set_usage(m.weights_buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

  std::ifstream fin(path,std::ios::binary);
  if(!fin){fprintf(stderr,"failed to reopen model weights: %s\n",path.c_str());free_model(m);return false;}
  std::vector<uint8_t> read_buffer;
  const size_t data_offset=gguf_get_data_offset(m.gguf);
  for(int64_t i=0;i<n_tensors;i++){
    const char*name=gguf_get_tensor_name(m.gguf,i);
    ggml_tensor*weight=m.t[name];
    const size_t size=ggml_nbytes(weight);
    const size_t offset=data_offset+gguf_get_tensor_offset(m.gguf,i);
    read_buffer.resize(size);
    fin.seekg((std::streamoff)offset,std::ios::beg);
    fin.read((char*)read_buffer.data(),(std::streamsize)size);
    if(!fin){fprintf(stderr,"failed to read model tensor: %s\n",name);free_model(m);return false;}
    ggml_backend_tensor_set(weight,read_buffer.data(),0,size);
  }
  return true;
}

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
  std::string gguf_path,fbank_path,wav_path,vad_path; int vad_maxseg=30000; bool ids_mode=false,keep_tags=false,srt_mode=false;
  std::string backend_name="cpu";
  for(int i=1;i<argc;i++){ if(!strcmp(argv[i],"-m")&&i+1<argc)gguf_path=argv[++i];
    else if(!strcmp(argv[i],"-f")&&i+1<argc)fbank_path=argv[++i];
    else if(!strcmp(argv[i],"-a")&&i+1<argc)wav_path=argv[++i];
    else if(!strcmp(argv[i],"--vad")&&i+1<argc)vad_path=argv[++i];
    else if(!strcmp(argv[i],"--vad-maxseg")&&i+1<argc)vad_maxseg=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--backend")&&i+1<argc)backend_name=argv[++i];
    else if(!strcmp(argv[i],"--ids"))ids_mode=true;
    else if(!strcmp(argv[i],"--keep-tags"))keep_tags=true;
    else if(!strcmp(argv[i],"--srt"))srt_mode=true;
    else {fprintf(stderr,"usage: %s -m sensevoice.gguf (-a audio.wav | -f fbank.bin) [--vad fsmn-vad.gguf [--vad-maxseg ms]] [--backend cpu|cuda|vulkan] [--srt] [--ids] [--keep-tags]\n",argv[0]);return 1;} }
  if(gguf_path.empty()||(fbank_path.empty()&&wav_path.empty())){fprintf(stderr,"missing args\n");return 1;}
  graph_backend graph_be=make_graph_backend(backend_name);

  // load model
  trace_stage("[sensevoice] loading model metadata");
  model m; if(!load_model_weights(gguf_path,graph_be.buffer_type,m))return 1; gguf_context*gg=m.gguf;
  auto rd=[&](const char*k,int d){int i=gguf_find_key(gg,k);return i<0?d:(int)gguf_get_val_u32(gg,i);};
  m.c.d_model=rd("sv.output_size",512); m.c.n_head=rd("sv.attention_heads",4);
  m.c.num_blocks=rd("sv.num_blocks",50); m.c.tp_blocks=rd("sv.tp_blocks",20);
  m.c.kernel=rd("sv.kernel_size",11); m.c.vocab=rd("sv.vocab_size",25055); m.c.blank=rd("sv.blank_id",0);
  int qi=gguf_find_key(gg,"sv.query_tokens"); int nq=qi<0?0:(int)gguf_get_arr_n(gg,qi);
  std::vector<int> qtok(nq); for(int i=0;i<nq;i++) qtok[i]=((const int32_t*)gguf_get_arr_data(gg,qi))[i];
  std::vector<std::string> vocab; {int ki=gguf_find_key(gg,"sv.vocab"); if(ki>=0){int nv=gguf_get_arr_n(gg,ki); vocab.resize(nv); for(int i=0;i<nv;i++){const char*s=gguf_get_arr_str(gg,ki,i); vocab[i]=s?s:"";}}}
  trace_stage("[sensevoice] model ready: %d tensors",gguf_get_n_tensors(gg));
  gguf_free(m.gguf); m.gguf=nullptr; ggml_free(m.ctx_meta); m.ctx_meta=nullptr;
  const int F=560, D=m.c.d_model, V=m.c.vocab;
  bool emit_ids = ids_mode || vocab.empty();   // fall back to ids if the gguf has no vocab

  // NOTE: SenseVoiceSmall inference() feeds the RAW log-mel fbank to the encoder;
  // it does NOT apply am.mvn CMVN (that path is unused at inference). Applying it
  // makes the encoder predict <|nospeech|>. So no CMVN here.
  ggml_tensor*embed=m.g("embed.weight");   // [16, 560] row-major
  if(embed->ne[0]!=F){fprintf(stderr,"unexpected embed width: %lld\n",(long long)embed->ne[0]);return 1;}
  for(int id:qtok) if(id<0||id>=embed->ne[1]){fprintf(stderr,"query token out of embed range: %d\n",id);return 1;}
  std::vector<float> embed_f32((size_t)ggml_nelements(embed));
  if(embed->type==GGML_TYPE_F32){
    ggml_backend_tensor_get(embed,embed_f32.data(),0,ggml_nbytes(embed));
  } else if(embed->type==GGML_TYPE_F16){
    std::vector<ggml_fp16_t> embed_f16(embed_f32.size());
    ggml_backend_tensor_get(embed,embed_f16.data(),0,ggml_nbytes(embed));
    for(size_t i=0;i<embed_f32.size();i++)embed_f32[i]=ggml_fp16_to_fp32(embed_f16[i]);
  } else {
    fprintf(stderr,"unsupported embed type: %s\n",ggml_type_name(embed->type));return 1;
  }
  // Run encoder+CTC on one fbank window [T,F]; returns decoded text string.
  auto run_seg=[&](const std::vector<float>& fb,int T) -> std::string {
    int N=nq+T; std::vector<float> inp((size_t)N*F);
    for(int i=0;i<nq;i++) memcpy(&inp[(size_t)i*F], &embed_f32[(size_t)qtok[i]*F], F*sizeof(float));
    memcpy(&inp[(size_t)nq*F], fb.data(), (size_t)T*F*sizeof(float));
    float sc=sqrtf((float)D); for(auto&v:inp)v*=sc; add_posenc(inp,N,F);
    trace_stage("[sensevoice] building graph: %d frames",N);
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
    trace_stage("[sensevoice] graph built");
    trace_stage("[sensevoice] allocating graph");
    ggml_gallocr_t ga=ggml_gallocr_new(graph_be.buffer_type); ggml_gallocr_alloc_graph(ga,gf);
    trace_stage("[sensevoice] graph allocated");
    ggml_backend_tensor_set(x,inp.data(),0,ggml_nbytes(x)); if(graph_be.is_cpu) ggml_backend_cpu_set_n_threads(graph_be.backend,8);
    trace_stage("[sensevoice] compute starting");
    enum ggml_status compute_status=ggml_backend_graph_compute(graph_be.backend,gf);
    trace_stage("[sensevoice] compute complete: status=%d",(int)compute_status);
    if(compute_status!=GGML_STATUS_SUCCESS){fprintf(stderr,"compute failed\n");}
    std::vector<float> lg((size_t)V*N); ggml_backend_tensor_get(logits,lg.data(),0,ggml_nbytes(logits));
    std::vector<int> seg_ids; int prev=-1;   // greedy CTC: argmax per frame -> collapse -> drop blank
    for(int n=0;n<N;n++){ const float*col=&lg[(size_t)n*V]; int am=0; float best=col[0];
      for(int v=1;v<V;v++) if(col[v]>best){best=col[v];am=v;}
      if(am!=prev && am!=m.c.blank) seg_ids.push_back(am); prev=am; }
    std::string result;
    if(emit_ids){ for(int id:seg_ids){ result+=std::to_string(id); result+=" "; } }
    else { result=detok_sv(seg_ids,vocab,keep_tags); }
    ggml_gallocr_free(ga); ggml_free(c);
    return result;
  };

  int64_t t0=ggml_time_us();
  int srt_idx=0;
  if(!vad_path.empty()){
    trace_stage("[sensevoice] loading audio");
    std::vector<float> wav; if(!funasr_load_audio_16k_mono(wav_path.c_str(),wav)){fprintf(stderr,"read audio failed\n");return 1;}
    trace_stage("[sensevoice] audio ready: %zu samples",wav.size());
    std::vector<std::pair<int,int>> segs;
    trace_stage("[sensevoice] running VAD");
    if(!funasr_vad_segments(vad_path,wav,vad_maxseg,segs)){fprintf(stderr,"vad failed\n");return 1;}
    trace_stage("[sensevoice] VAD ready: %zu segments",segs.size());
    for(auto&s:segs){ int off=(int)((int64_t)s.first*16000/1000), end=(int)((int64_t)s.second*16000/1000);
      if(end>(int)wav.size())end=wav.size(); if(end-off<WINLEN)continue;
      std::vector<float> seg(wav.begin()+off,wav.begin()+end); int t=0; auto fb=compute_fbank(seg,t);
      std::string text=run_seg(fb,t);
      if(text.empty())continue;
      if(srt_mode){ srt_idx++; format_srt_line(srt_idx,s.first,s.second,text); }
      else { printf("%s",text.c_str()); }
      fflush(stdout);
    }
    fprintf(stderr,"[sensevoice] %zu vad segments\n",segs.size());
  } else {
    int32_t T=0,Fc=F; std::vector<float> fb; int end_ms=0;
    if(!wav_path.empty()){
      trace_stage("[sensevoice] loading audio");
      std::vector<float> wav; if(!funasr_load_audio_16k_mono(wav_path.c_str(),wav)){fprintf(stderr,"read audio failed\n");return 1;}
      trace_stage("[sensevoice] audio ready: %zu samples",wav.size());
      end_ms=(int)((int64_t)wav.size()*1000/16000);
      int t=0; fb=compute_fbank(wav,t); T=t;
    } else {
      FILE*f=fopen(fbank_path.c_str(),"rb"); if(!f){fprintf(stderr,"open fbank\n");return 1;}
      if(fread(&T,4,1,f)!=1||fread(&Fc,4,1,f)!=1){fclose(f);return 1;}
      fb.resize((size_t)T*Fc); if((int)fread(fb.data(),4,fb.size(),f)!=(int)fb.size()){fclose(f);return 1;} fclose(f);
      end_ms=(int)((int64_t)T*LFR_N*SHIFT*1000/FS);
    }
    std::string text=run_seg(fb,T);
    if(srt_mode){ if(!text.empty()) format_srt_line(1,0,end_ms,text); }
    else { printf("%s",text.c_str()); }
    fflush(stdout);
  }
  if(!srt_mode) printf("\n");
  fprintf(stderr,"[sensevoice] done %.2fs\n",(ggml_time_us()-t0)/1e6);
  free_model(m);
  ggml_backend_free(graph_be.backend);
  return 0;
}

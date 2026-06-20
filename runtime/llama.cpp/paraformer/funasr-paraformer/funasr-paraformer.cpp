// funasr-paraformer: Paraformer (non-autoregressive ASR) on ggml.
//   WAV → kaldi fbank → CMVN → SANM encoder (ggml) → CIF predictor (host) →
//   SANM decoder w/ cross-attn (ggml) → argmax → token ids (stdout).
// Encoder/FSMN/attention primitives are shared with the Fun-ASR-Nano runtime.

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

// ===== audio loader: any wav/mp3/flac, any rate/channels -> 16k mono (miniaudio) =====
#define FUNASR_AUDIO_IMPLEMENTATION
#include "funasr_audio.h"
#include "funasr_vad.h"     // built-in FSMN-VAD front end (--vad segmentation)
#include <utility>
static const int FS=16000,WINLEN=400,SHIFT=160,NFFT=512,NMEL=80,LFR_M=7,LFR_N=6;
static const float PREEMPH=0.97f,LOWF=20.0f,HIGHF=8000.0f;
static inline float melf(float f){return 1127.0f*logf(1.0f+f/700.0f);}
static void fftc(std::vector<float>&re,std::vector<float>&im,int n){for(int i=1,j=0;i<n;i++){int b=n>>1;for(;j&b;b>>=1)j^=b;j^=b;if(i<j){std::swap(re[i],re[j]);std::swap(im[i],im[j]);}}
  for(int len=2;len<=n;len<<=1){double a=-2.0*M_PI/len;float wr=cosf(a),wi=sinf(a);for(int i=0;i<n;i+=len){float cr=1,ci=0;for(int k=0;k<len/2;k++){float ur=re[i+k],ui=im[i+k];
    float vr=re[i+k+len/2]*cr-im[i+k+len/2]*ci,vi=re[i+k+len/2]*ci+im[i+k+len/2]*cr;re[i+k]=ur+vr;im[i+k]=ui+vi;re[i+k+len/2]=ur-vr;im[i+k+len/2]=ui-vi;float nc=cr*wr-ci*wi;ci=cr*wi+ci*wr;cr=nc;}}}}
static std::vector<float> compute_fbank(std::vector<float> wav,int&T_out){
  for(auto&v:wav)v*=32768.0f;std::vector<float>win(WINLEN);for(int i=0;i<WINLEN;i++)win[i]=0.54f-0.46f*cosf(2.0f*M_PI*i/(WINLEN-1));
  const int NB=NFFT/2+1;float bw=(float)FS/NFFT,ml=melf(LOWF),mh=melf(HIGHF),dm=(mh-ml)/(NMEL+1);
  std::vector<std::vector<float>>fb(NMEL,std::vector<float>(NB,0.0f));
  for(int m=0;m<NMEL;m++){float L=ml+m*dm,C=ml+(m+1)*dm,R=ml+(m+2)*dm;for(int k=0;k<NB;k++){float mf=melf(bw*k);if(mf>L&&mf<R)fb[m][k]=mf<=C?(mf-L)/(C-L):(R-mf)/(R-C);}}
  int N=wav.size(),T=(N-WINLEN)/SHIFT+1;std::vector<std::vector<float>>feat(T,std::vector<float>(NMEL));
  std::vector<float>re(NFFT),im(NFFT),fr(WINLEN);const float fl=1.1920929e-07f;
  for(int t=0;t<T;t++){const float*s=wav.data()+t*SHIFT;double mn=0;for(int i=0;i<WINLEN;i++)mn+=s[i];mn/=WINLEN;
    for(int i=0;i<WINLEN;i++)fr[i]=s[i]-(float)mn;for(int i=WINLEN-1;i>0;i--)fr[i]-=PREEMPH*fr[i-1];fr[0]-=PREEMPH*fr[0];
    for(int i=0;i<NFFT;i++){re[i]=i<WINLEN?fr[i]*win[i]:0.0f;im[i]=0.0f;}fftc(re,im,NFFT);
    for(int m=0;m<NMEL;m++){float e=0;for(int k=0;k<NB;k++)if(fb[m][k]>0)e+=fb[m][k]*(re[k]*re[k]+im[k]*im[k]);feat[t][m]=logf(e>fl?e:fl);}}
  const int pad=(LFR_M-1)/2;int Tl=(T+LFR_N-1)/LFR_N;std::vector<std::vector<float>>pd;pd.reserve(T+pad+LFR_M);
  for(int i=0;i<pad;i++)pd.push_back(feat[0]);for(int t=0;t<T;t++)pd.push_back(feat[t]);while((int)pd.size()<(Tl-1)*LFR_N+LFR_M)pd.push_back(feat[T-1]);
  int D=LFR_M*NMEL;std::vector<float>out((size_t)Tl*D);for(int i=0;i<Tl;i++)for(int j=0;j<LFR_M;j++)memcpy(&out[(size_t)i*D+j*NMEL],pd[i*LFR_N+j].data(),NMEL*sizeof(float));
  T_out=Tl;return out;}

// ===== model =====
struct cfg{int d_model=512,enc_head=4,enc_blocks=50,enc_kernel=11,dec_blocks=16,dec_att=16,dec3=1,dec_head=4,dec_kernel=11,vocab=8404;float tail=0.45f,thresh=1.0f;};
struct model{cfg c;ggml_context*ctx_w=nullptr;std::map<std::string,ggml_tensor*>t;
  ggml_tensor*g(const std::string&n){auto it=t.find(n);if(it==t.end()){fprintf(stderr,"missing %s\n",n.c_str());exit(1);}return it->second;}
  bool has(const std::string&n){return t.count(n);} };

static ggml_tensor* lin(ggml_context*c,ggml_tensor*w,ggml_tensor*b,ggml_tensor*x){auto y=ggml_mul_mat(c,w,x);return b?ggml_add(c,y,b):y;}
static ggml_tensor* lnorm(ggml_context*c,ggml_tensor*x,ggml_tensor*g,ggml_tensor*b){return ggml_add(c,ggml_mul(c,ggml_norm(c,x,LN_EPS),g),b);}
// FSMN depthwise (kernel stored [K,D]): out = sum_j w[:,j]*pad(v)[:, t+j]  (+ residual v)
static ggml_tensor* fsmn(ggml_context*c,ggml_tensor*v,ggml_tensor*fk,int D,int T,int K){
  const int pad=(K-1)/2;ggml_tensor*vp=ggml_pad_ext(c,v,0,0,pad,pad,0,0,0,0);ggml_tensor*acc=v;
  for(int j=0;j<K;j++){auto sl=ggml_view_2d(c,vp,D,T,vp->nb[1],(size_t)j*vp->nb[1]);auto wj=ggml_view_1d(c,fk,D,(size_t)j*fk->nb[1]);
    acc=ggml_add(c,acc,ggml_mul(c,ggml_cont(c,sl),wj));}return acc;}
// SAN-M encoder self-attn (fused qkv + fsmn)
static ggml_tensor* enc_attn(ggml_context*c,model&m,const std::string&p,ggml_tensor*x,int T){
  const int D=m.c.d_model,H=m.c.enc_head,dk=D/H,K=m.c.enc_kernel;
  ggml_tensor*qkv=lin(c,m.g(p+"linear_q_k_v.weight"),m.g(p+"linear_q_k_v.bias"),x);size_t nb1=qkv->nb[1];
  ggml_tensor*q=ggml_cont(c,ggml_view_2d(c,qkv,D,T,nb1,0));
  ggml_tensor*k=ggml_cont(c,ggml_view_2d(c,qkv,D,T,nb1,(size_t)D*sizeof(float)));
  ggml_tensor*v=ggml_cont(c,ggml_view_2d(c,qkv,D,T,nb1,(size_t)2*D*sizeof(float)));
  ggml_tensor*fm=fsmn(c,v,m.g(p+"fsmn_block.weight"),D,T,K);
  q=ggml_permute(c,ggml_reshape_3d(c,q,dk,H,T),0,2,1,3);k=ggml_permute(c,ggml_reshape_3d(c,k,dk,H,T),0,2,1,3);
  ggml_tensor*vh=ggml_cont(c,ggml_permute(c,ggml_reshape_3d(c,v,dk,H,T),1,2,0,3));
  ggml_tensor*kq=ggml_soft_max(c,ggml_scale(c,ggml_mul_mat(c,k,q),1.0f/sqrtf((float)dk)));
  ggml_tensor*o=ggml_cont_2d(c,ggml_permute(c,ggml_mul_mat(c,vh,kq),0,2,1,3),D,T);
  return ggml_add(c,lin(c,m.g(p+"linear_out.weight"),m.g(p+"linear_out.bias"),o),fm);}
static ggml_tensor* enc_layer(ggml_context*c,model&m,const std::string&p,ggml_tensor*x,int T,bool res){
  auto r=x;auto h=lnorm(c,x,m.g(p+"norm1.weight"),m.g(p+"norm1.bias"));auto sa=enc_attn(c,m,p+"self_attn.",h,T);
  x=res?ggml_add(c,r,sa):sa;r=x;h=lnorm(c,x,m.g(p+"norm2.weight"),m.g(p+"norm2.bias"));
  h=lin(c,m.g(p+"feed_forward.w_1.weight"),m.g(p+"feed_forward.w_1.bias"),h);h=ggml_relu(c,h);
  h=lin(c,m.g(p+"feed_forward.w_2.weight"),m.g(p+"feed_forward.w_2.bias"),h);return ggml_add(c,r,h);}
// decoder FFN-SANM: w_2(LayerNorm(relu(w_1(x))))   (w_2 has no bias)
static ggml_tensor* dec_ffn(ggml_context*c,model&m,const std::string&p,ggml_tensor*x){
  auto h=lin(c,m.g(p+"w_1.weight"),m.g(p+"w_1.bias"),x);h=ggml_relu(c,h);
  h=lnorm(c,h,m.g(p+"norm.weight"),m.g(p+"norm.bias"));return ggml_mul_mat(c,m.g(p+"w_2.weight"),h);}
// cross attn: q=linear_q(tgt)[D,N], kv=linear_k_v(mem)[2D,T]
static ggml_tensor* cross_attn(ggml_context*c,model&m,const std::string&p,ggml_tensor*tgt,ggml_tensor*mem,int N,int T){
  const int D=m.c.d_model,H=m.c.dec_head,dk=D/H;
  ggml_tensor*q=lin(c,m.g(p+"linear_q.weight"),m.g(p+"linear_q.bias"),tgt);     // [D,N]
  ggml_tensor*kv=lin(c,m.g(p+"linear_k_v.weight"),m.g(p+"linear_k_v.bias"),mem);// [2D,T]
  size_t nb1=kv->nb[1];
  ggml_tensor*k=ggml_cont(c,ggml_view_2d(c,kv,D,T,nb1,0));
  ggml_tensor*v=ggml_cont(c,ggml_view_2d(c,kv,D,T,nb1,(size_t)D*sizeof(float)));
  q=ggml_permute(c,ggml_reshape_3d(c,q,dk,H,N),0,2,1,3);   // [dk,N,H]
  k=ggml_permute(c,ggml_reshape_3d(c,k,dk,H,T),0,2,1,3);   // [dk,T,H]
  ggml_tensor*vh=ggml_cont(c,ggml_permute(c,ggml_reshape_3d(c,v,dk,H,T),1,2,0,3)); // [T,dk,H]
  ggml_tensor*kq=ggml_soft_max(c,ggml_scale(c,ggml_mul_mat(c,k,q),1.0f/sqrtf((float)dk))); // [T,N,H]
  ggml_tensor*o=ggml_cont_2d(c,ggml_permute(c,ggml_mul_mat(c,vh,kq),0,2,1,3),D,N);
  return lin(c,m.g(p+"linear_out.weight"),m.g(p+"linear_out.bias"),o);}
static ggml_tensor* dec_layer(ggml_context*c,model&m,const std::string&p,ggml_tensor*tgt,ggml_tensor*mem,int N,int T){
  const int D=m.c.d_model,K=m.c.dec_kernel;
  auto residual=tgt;auto h=lnorm(c,tgt,m.g(p+"norm1.weight"),m.g(p+"norm1.bias"));
  h=dec_ffn(c,m,p+"feed_forward.",h);                       // FFN first
  auto y=lnorm(c,h,m.g(p+"norm2.weight"),m.g(p+"norm2.bias"));
  auto sa=fsmn(c,y,m.g(p+"self_attn.fsmn_block.weight"),D,N,K);  // FSMN self-attn (+residual y inside)
  auto x=ggml_add(c,residual,sa);
  residual=x;auto z=lnorm(c,x,m.g(p+"norm3.weight"),m.g(p+"norm3.bias"));
  auto ca=cross_attn(c,m,p+"src_attn.",z,mem,N,T);
  return ggml_add(c,residual,ca);}
static ggml_tensor* dec3_layer(ggml_context*c,model&m,const std::string&p,ggml_tensor*tgt){
  auto h=lnorm(c,tgt,m.g(p+"norm1.weight"),m.g(p+"norm1.bias"));return dec_ffn(c,m,p+"feed_forward.",h);}
static void add_posenc(std::vector<float>&x,int T,int depth){double inc=log(10000.0)/(depth/2.0-1.0);
  for(int t=0;t<T;t++){double pos=t+1;for(int i=0;i<depth/2;i++){double its=exp(i*-inc),st=pos*its;x[(size_t)t*depth+i]+=(float)sin(st);x[(size_t)t*depth+depth/2+i]+=(float)cos(st);}}}

// run a ggml graph on CPU, return output [ne0 x ne1] row-major (ne1 rows)
static std::vector<float> run_graph(ggml_context*c,ggml_tensor*out,ggml_tensor*in1,const float*d1,ggml_tensor*in2,const float*d2){
  ggml_backend_t be=ggml_backend_cpu_init();ggml_cgraph*gf=ggml_new_graph_custom(c,32768,false);ggml_build_forward_expand(gf,out);
  ggml_gallocr_t ga=ggml_gallocr_new(ggml_backend_cpu_buffer_type());ggml_gallocr_alloc_graph(ga,gf);
  ggml_backend_tensor_set(in1,d1,0,ggml_nbytes(in1));if(in2)ggml_backend_tensor_set(in2,d2,0,ggml_nbytes(in2));
  ggml_backend_cpu_set_n_threads(be,8);ggml_backend_graph_compute(be,gf);
  int D=out->ne[0],N=out->ne[1];std::vector<float>r((size_t)D*N);ggml_backend_tensor_get(out,r.data(),0,ggml_nbytes(out));
  ggml_gallocr_free(ga);ggml_backend_free(be);return r;}

// Paraformer detok: tokens.json is BPE; join, drop "@@" continuations, "▁"(U+2581)->space.
static std::string pf_trim(const std::string&s){size_t a=s.find_first_not_of(' ');if(a==std::string::npos)return "";size_t b=s.find_last_not_of(' ');return s.substr(a,b-a+1);}
static std::string detok_pf(const std::vector<int>&ids,const std::vector<std::string>&vocab){
  std::string s; for(int id:ids){ if(id==1||id==2)continue; if(id>=0&&id<(int)vocab.size())s+=vocab[id]; }
  size_t p; while((p=s.find("@@"))!=std::string::npos)s.erase(p,2);
  const std::string lb="\xe2\x96\x81"; while((p=s.find(lb))!=std::string::npos)s.replace(p,3," ");
  return pf_trim(s);
}

int main(int argc,char**argv){
  std::string gguf_path,wav_path,vad_path; int vad_maxseg=30000; bool ids_mode=false;
  for(int i=1;i<argc;i++){if(!strcmp(argv[i],"-m")&&i+1<argc)gguf_path=argv[++i];else if(!strcmp(argv[i],"-a")&&i+1<argc)wav_path=argv[++i];
    else if(!strcmp(argv[i],"--vad")&&i+1<argc)vad_path=argv[++i];
    else if(!strcmp(argv[i],"--vad-maxseg")&&i+1<argc)vad_maxseg=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--ids"))ids_mode=true;
    else{fprintf(stderr,"usage: %s -m paraformer.gguf -a audio.wav [--vad fsmn-vad.gguf [--vad-maxseg ms]] [--ids]\n",argv[0]);return 1;}}
  if(gguf_path.empty()||wav_path.empty()){fprintf(stderr,"missing args\n");return 1;}
  model m;gguf_init_params gp={false,&m.ctx_w};gguf_context*gg=gguf_init_from_file(gguf_path.c_str(),gp);if(!gg){fprintf(stderr,"gguf load failed\n");return 1;}
  auto rdi=[&](const char*k,int d){int i=gguf_find_key(gg,k);return i<0?d:(int)gguf_get_val_u32(gg,i);};
  auto rdf=[&](const char*k,float d){int i=gguf_find_key(gg,k);return i<0?d:gguf_get_val_f32(gg,i);};
  m.c.enc_blocks=rdi("pf.enc.num_blocks",50);m.c.dec_blocks=rdi("pf.dec.num_blocks",16);m.c.dec_att=rdi("pf.dec.att_layer_num",16);
  m.c.dec3=rdi("pf.dec.decoders3",1);m.c.vocab=rdi("pf.vocab_size",8404);m.c.tail=rdf("pf.predictor.tail_threshold",0.45f);m.c.thresh=rdf("pf.predictor.threshold",1.0f);
  std::vector<std::string> vocab; {int ki=gguf_find_key(gg,"pf.vocab"); if(ki>=0){int nv=gguf_get_arr_n(gg,ki); vocab.resize(nv); for(int i=0;i<nv;i++){const char*s=gguf_get_arr_str(gg,ki,i); vocab[i]=s?s:"";}}}
  for(int i=0;i<gguf_get_n_tensors(gg);i++){const char*nm=gguf_get_tensor_name(gg,i);m.t[nm]=ggml_get_tensor(m.ctx_w,nm);}gguf_free(gg);
  const int D=m.c.d_model,F=560,V=m.c.vocab;
  bool emit_ids = ids_mode || vocab.empty();   // fall back to ids if the gguf has no vocab

  float*shift=(float*)m.g("cmvn.shift")->data,*scale=(float*)m.g("cmvn.scale")->data;
  float*cw=(float*)m.g("predictor.cif_conv1d.weight")->data; // [512,512,3] = [o][i][j]
  float*cb=(float*)m.g("predictor.cif_conv1d.bias")->data;   // [512]
  float*ow=(float*)m.g("predictor.cif_output.weight")->data; // [1,512]
  float ob=((float*)m.g("predictor.cif_output.bias")->data)[0];

  // Full pipeline (fbank -> CMVN -> encoder -> CIF -> decoder) on one wav window; prints token IDs.
  auto run_seg=[&](const std::vector<float>& wav){
    int T=0;auto fb=compute_fbank(wav,T); if(T<1)return;
    for(int t=0;t<T;t++)for(int d=0;d<F;d++)fb[(size_t)t*F+d]=(fb[(size_t)t*F+d]+shift[d])*scale[d];  // CMVN
    {float sc=sqrtf((float)D);for(auto&v:fb)v*=sc;}add_posenc(fb,T,F);
    std::vector<float>enc;
    {ggml_init_params cp={(size_t)1024*1024*1024,nullptr,true};ggml_context*c=ggml_init(cp);
     ggml_tensor*x=ggml_new_tensor_2d(c,GGML_TYPE_F32,F,T);ggml_set_input(x);
     ggml_tensor*h=enc_layer(c,m,"encoder.encoders0.0.",x,T,false);
     for(int i=0;i<m.c.enc_blocks-1;i++)h=enc_layer(c,m,"encoder.encoders."+std::to_string(i)+".",h,T,true);
     h=lnorm(c,h,m.g("encoder.after_norm.weight"),m.g("encoder.after_norm.bias"));ggml_set_output(h);
     enc=run_graph(c,h,x,fb.data(),nullptr,nullptr);ggml_free(c);}   // enc row-major [T,D]
    // ===== CIF predictor (host): conv1d(k3,pad1)+residual+relu -> cif_output -> sigmoid -> alpha =====
    std::vector<float> outp((size_t)T*D); std::vector<float> alphas(T);
    for(int t=0;t<T;t++){
      for(int o=0;o<D;o++){ float acc=cb[o];
        for(int j=0;j<3;j++){int tt=t+j-1; if(tt<0||tt>=T)continue; const float*ev=&enc[(size_t)tt*D];
          const float*wo=&cw[(size_t)o*D*3]; for(int i=0;i<D;i++) acc+=wo[i*3+j]*ev[i];}
        outp[(size_t)t*D+o]=acc+enc[(size_t)t*D+o]; }
      float a=ob; for(int o=0;o<D;o++){float r=outp[(size_t)t*D+o]; if(r<0)r=0; a+=ow[o]*r;}
      float s=1.0f/(1.0f+expf(-a)); alphas[t]=s>0?s:0; }
    std::vector<float> hid=enc; hid.resize((size_t)(T+1)*D,0.0f);
    std::vector<float> al=alphas; al.push_back(m.c.tail); int L=T+1;
    std::vector<float> acoustic; acoustic.reserve(64*D);
    float integrate=0; std::vector<float> frame(D,0.0f);
    for(int t=0;t<L;t++){
      float alpha=al[t]; float dc=1.0f-integrate; integrate+=alpha;
      bool fire=integrate>=m.c.thresh; float cur=fire?dc:alpha; float rem=alpha-cur;
      for(int d=0;d<D;d++) frame[d]+=cur*hid[(size_t)t*D+d];
      if(fire){ acoustic.insert(acoustic.end(),frame.begin(),frame.end()); integrate-=1.0f;
        for(int d=0;d<D;d++) frame[d]=rem*hid[(size_t)t*D+d]; } }
    int N=acoustic.size()/D; if(N<1)return;
    std::vector<float> logits;
    {ggml_init_params cp={(size_t)2048*1024*1024,nullptr,true};ggml_context*c=ggml_init(cp);
     ggml_tensor*tgt=ggml_new_tensor_2d(c,GGML_TYPE_F32,D,N);ggml_set_input(tgt);
     ggml_tensor*mem=ggml_new_tensor_2d(c,GGML_TYPE_F32,D,T);ggml_set_input(mem);
     ggml_tensor*x=tgt;
     for(int i=0;i<m.c.dec_att;i++)x=dec_layer(c,m,"decoder.decoders."+std::to_string(i)+".",x,mem,N,T);
     for(int i=0;i<m.c.dec3;i++)x=dec3_layer(c,m,"decoder.decoders3."+std::to_string(i)+".",x);
     x=lnorm(c,x,m.g("decoder.after_norm.weight"),m.g("decoder.after_norm.bias"));
     x=lin(c,m.g("decoder.output_layer.weight"),m.g("decoder.output_layer.bias"),x); // [V,N]
     ggml_set_output(x);
     logits=run_graph(c,x,tgt,acoustic.data(),mem,enc.data());ggml_free(c);}
    std::vector<int> seg_ids; seg_ids.reserve(N);
    for(int n=0;n<N;n++){const float*col=&logits[(size_t)n*V];int am=0;float best=col[0];for(int v=1;v<V;v++)if(col[v]>best){best=col[v];am=v;}seg_ids.push_back(am);}
    if(emit_ids){ for(int id:seg_ids) printf("%d ",id); }
    else { std::string t=detok_pf(seg_ids,vocab); printf("%s",t.c_str()); }
  };

  int64_t t0=ggml_time_us();
  std::vector<float>wav;if(!funasr_load_audio_16k_mono(wav_path.c_str(),wav)){fprintf(stderr,"read audio failed\n");return 1;}
  if(!vad_path.empty()){
    std::vector<std::pair<int,int>> segs;
    if(!funasr_vad_segments(vad_path,wav,vad_maxseg,segs)){fprintf(stderr,"vad failed\n");return 1;}
    for(auto&s:segs){ int off=(int)((int64_t)s.first*16000/1000), end=(int)((int64_t)s.second*16000/1000);
      if(end>(int)wav.size())end=wav.size(); if(end-off<WINLEN)continue;
      std::vector<float> seg(wav.begin()+off,wav.begin()+end); run_seg(seg); }
    fprintf(stderr,"[paraformer] %zu vad segments\n",segs.size());
  } else { run_seg(wav); }
  printf("\n");
  fprintf(stderr,"[paraformer] done %.2fs\n",(ggml_time_us()-t0)/1e6);
  if(m.ctx_w) ggml_free(m.ctx_w);
  return 0;
}

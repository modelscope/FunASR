/*
pcm编码器+编码引擎
https://github.com/xiangyuecn/Recorder

编码原理：本编码器输出的pcm格式数据其实就是Recorder中的buffers原始数据（经过了重新采样），16位时为LE小端模式（Little Endian），并未经过任何编码处理

编码的代码和wav.js区别不大，pcm加上一个44字节wav头即成wav文件；所以要播放pcm就很简单了，直接转成wav文件来播放，已提供转换函数 Recorder.pcm2wav
*/
(function(){
"use strict";

Recorder.prototype.enc_pcm={
	stable:true
	,testmsg:"pcm为未封装的原始音频数据，pcm数据文件无法直接播放；支持位数8位、16位（填在比特率里面），采样率取值无限制"
};
Recorder.prototype.pcm=function(res,True,False){
		var This=this,set=This.set
			,size=res.length
			,bitRate=set.bitRate==8?8:16;
		
		var buffer=new ArrayBuffer(size*(bitRate/8));
		var data=new DataView(buffer);
		var offset=0;
		
		// 写入采样数据
		if(bitRate==8) {
			for(var i=0;i<size;i++,offset++) {
				//16转8据说是雷霄骅的 https://blog.csdn.net/sevennight1989/article/details/85376149 细节比blqw的按比例的算法清晰点，虽然都有明显杂音
				var val=(res[i]>>8)+128;
				data.setInt8(offset,val,true);
			};
		}else{
			for (var i=0;i<size;i++,offset+=2){
				data.setInt16(offset,res[i],true);
			};
		};
		
		
		True(new Blob([data.buffer],{type:"audio/pcm"}));
	};





/**pcm直接转码成wav，可以直接用来播放；需同时引入wav.js
data: {
		sampleRate:16000 pcm的采样率
		bitRate:16 pcm的位数 取值：8 或 16
		blob:blob对象
	}
	data如果直接提供的blob将默认使用16位16khz的配置，仅用于测试
True(wavBlob,duration)
False(msg)
**/
Recorder.pcm2wav=function(data,True,False){
	if(data.slice && data.type!=null){//Blob 测试用
		data={blob:data};
	};
	var sampleRate=data.sampleRate||16000,bitRate=data.bitRate||16;
	if(!data.sampleRate || !data.bitRate){
		console.warn("pcm2wav必须提供sampleRate和bitRate");
	};
	if(!Recorder.prototype.wav){
		False("pcm2wav必须先加载wav编码器wav.js");
		return;
	};
	
	var reader=new FileReader();
	reader.onloadend=function(){
		var pcm;
		if(bitRate==8){
			//8位转成16位
			var u8arr=new Uint8Array(reader.result);
			pcm=new Int16Array(u8arr.length);
			for(var j=0;j<u8arr.length;j++){
				pcm[j]=(u8arr[j]-128)<<8;
			};
		}else{
			pcm=new Int16Array(reader.result);
		};
		
		Recorder({
			type:"wav"
			,sampleRate:sampleRate
			,bitRate:bitRate
		}).mock(pcm,sampleRate).stop(function(wavBlob,duration){
			True(wavBlob,duration);
		},False);
	};
	reader.readAsArrayBuffer(data.blob);
};



})();
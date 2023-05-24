/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
/* 2022-2023 by zhaoming,mali aihealthx.com */


// 连接; 定义socket连接类对象与语音对象
var wsconnecter = new WebSocketConnectMethod({msgHandle:getJsonMessage,stateHandle:getConnState});
var audioBlob;

// 录音; 定义录音对象,wav格式
var rec = Recorder({
	type:"pcm",
	bitRate:16,
	sampleRate:16000,
	onProcess:recProcess
});

 
 
 
var sampleBuf=new Int16Array();
// 定义按钮响应事件
var btnStart = document.getElementById('btnStart');
btnStart.onclick = start;
var btnStop = document.getElementById('btnStop');
btnStop.onclick = stop;
btnStop.disabled = true;
 

 
var rec_text=""
var info_div = document.getElementById('info_div');

//var now_ipaddress=window.location.href;
//now_ipaddress=now_ipaddress.replace("https://","wss://");
//now_ipaddress=now_ipaddress.replace("static/index.html","");
//document.getElementById('wssip').value=now_ipaddress;

// 语音识别结果; 对jsonMsg数据解析,将识别结果附加到编辑框中
function getJsonMessage( jsonMsg ) {
	console.log( "message: " + JSON.parse(jsonMsg.data)['text'] );
	var rectxt=""+JSON.parse(jsonMsg.data)['text'];
	var varArea=document.getElementById('varArea');
	rec_text=rec_text+rectxt.replace(/ +/g,"");
	varArea.value=rec_text;
	 
 
}

// 连接状态响应
function getConnState( connState ) {
	if ( connState === 0 ) {
 
		rec.open( function(){
			rec.start();
			console.log("开始录音");
 
		});
	} else if ( connState === 1 ) {
		//stop();
	} else if ( connState === 2 ) {
		stop();
		console.log( 'connecttion error' );
		 
		alert("连接地址"+document.getElementById('wssip').value+"失败,请检查asr地址和端口，并确保h5服务和asr服务在同一个域内。或换个浏览器试试。");
		btnStart.disabled = true;
		info_div.innerHTML='请点击开始';
	}
}


// 识别启动、停止、清空操作
function start() {
	
	// 清除显示
	clear();
	//控件状态更新
 	    

	//启动连接
	var ret=wsconnecter.wsStart();
	if(ret==1){
		isRec = true;
		btnStart.disabled = true;
		btnStop.disabled = false;
	    info_div.innerHTML="正在连接asr服务器，请等待...";
	}
}

 
function stop() {
		var chunk_size = new Array( 5, 10, 5 );
		var request = {
			"chunk_size": chunk_size,
			"wav_name":  "h5",
			"is_speaking":  false,
			"chunk_interval":10,
		};
		if(sampleBuf.length>0){
		wsconnecter.wsSend(sampleBuf,false);
		console.log("sampleBuf.length"+sampleBuf.length);
		sampleBuf=new Int16Array();
		}
	   wsconnecter.wsSend( JSON.stringify(request) ,false);
 
	 
	
	

	
	// 控件状态更新
	isRec = false;
    info_div.innerHTML="请等候...";
	btnStop.disabled = true;
	setTimeout(function(){btnStart.disabled = false;info_div.innerHTML="请点击开始";}, 3000 );
	rec.stop(function(blob,duration){
  
		console.log(blob);
		var audioBlob = Recorder.pcm2wav(data = {sampleRate:16000, bitRate:16, blob:blob},
		function(theblob,duration){
				console.log(theblob);
		var audio_record = document.getElementById('audio_record');
		audio_record.src =  (window.URL||webkitURL).createObjectURL(theblob); 
        audio_record.controls=true;
		audio_record.play(); 
         	

	}   ,function(msg){
		 console.log(msg);
	}
		);
 

 
	},function(errMsg){
		console.log("errMsg: " + errMsg);
	});
    // 停止连接
	
    

}

function clear() {
 
    var varArea=document.getElementById('varArea');
 
	varArea.value="";
    rec_text="";
 
}

 
function recProcess( buffer, powerLevel, bufferDuration, bufferSampleRate,newBufferIdx,asyncEnd ) {
	if ( isRec === true ) {
		var data_48k = buffer[buffer.length-1];  
 
		var  array_48k = new Array(data_48k);
		var data_16k=Recorder.SampleData(array_48k,bufferSampleRate,16000).data;
 
		sampleBuf = Int16Array.from([...sampleBuf, ...data_16k]);
		var chunk_size=960; // for asr chunk_size [5, 10, 5]
		info_div.innerHTML=""+bufferDuration/1000+"s";
		while(sampleBuf.length>=chunk_size){
		    sendBuf=sampleBuf.slice(0,chunk_size);
			sampleBuf=sampleBuf.slice(chunk_size,sampleBuf.length);
			wsconnecter.wsSend(sendBuf,false);
			
			
		 
		}
		
 
		
	}
}
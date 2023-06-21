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
btnStart.onclick = record;
var btnStop = document.getElementById('btnStop');
btnStop.onclick = stop;
btnStop.disabled = true;
btnStart.disabled = true;
 
btnConnect= document.getElementById('btnConnect');
btnConnect.onclick = start;
 
var rec_text="";  // for online rec asr result
var offline_text=""; // for offline rec asr result
var info_div = document.getElementById('info_div');

var upfile = document.getElementById('upfile');

 

var isfilemode=false;  // if it is in file mode
var file_data_array;  // array to save file data
var isconnected=0;    // for file rec, 0 is not begin, 1 is connected, -1 is error
var totalsend=0;
 
upfile.onchange = function () {
　　　　　　var len = this.files.length;  
            for(let i = 0; i < len; i++) {
                let fileAudio = new FileReader();
                fileAudio.readAsArrayBuffer(this.files[i]);  
                fileAudio.onload = function() {
                 var audioblob= fileAudio.result;
				 file_data_array=audioblob;
				 console.log(audioblob);
                 btnConnect.disabled = false;
                 info_div.innerHTML='请点击连接进行识别';
               
                }
　　　　　　　　　　fileAudio.onerror = function(e) {
　　　　　　　　　　　　console.log('error' + e);
　　　　　　　　　　}
            }
        }

function play_file()
{
		  var audioblob=new Blob( [ new Uint8Array(file_data_array)] , {type :"audio/wav"});
		  var audio_record = document.getElementById('audio_record');
		  audio_record.src =  (window.URL||webkitURL).createObjectURL(audioblob); 
          audio_record.controls=true;
		  audio_record.play(); 
}
function start_file_send()
{
		sampleBuf=new Int16Array( file_data_array );
 
		var chunk_size=960; // for asr chunk_size [5, 10, 5]
 

 
		
 
		while(sampleBuf.length>=chunk_size){
			
		    sendBuf=sampleBuf.slice(0,chunk_size);
			totalsend=totalsend+sampleBuf.length;
			sampleBuf=sampleBuf.slice(chunk_size,sampleBuf.length);
			wsconnecter.wsSend(sendBuf,false);
 
		 
		}
 
		stop();

 

}
function start_file_offline()
{             
           	  console.log("start_file_offline",isconnected);  
              if(isconnected==-1)
			  {
				  return;
			  }
		      if(isconnected==0){
			   
		        setTimeout(start_file_offline, 1000);
				return;
		      }
			start_file_send();
 
	         

		 
}
	
function on_recoder_mode_change()
{
            var item = null;
            var obj = document.getElementsByName("recoder_mode");
            for (var i = 0; i < obj.length; i++) { //遍历Radio 
                if (obj[i].checked) {
                    item = obj[i].value;  
					break;
                }
		    

           }
		    if(item=="mic")
			{
				document.getElementById("mic_mode_div").style.display = 'block';
				document.getElementById("rec_mode_div").style.display = 'none';
 
				btnConnect.disabled=false;
				isfilemode=false;
			}
			else
			{
				document.getElementById("mic_mode_div").style.display = 'none';
				document.getElementById("rec_mode_div").style.display = 'block';
                btnConnect.disabled = true;
			    isfilemode=true;
				info_div.innerHTML='请点击选择文件';
			    
	 
			}
}
function getAsrMode(){

            var item = null;
            var obj = document.getElementsByName("asr_mode");
            for (var i = 0; i < obj.length; i++) { //遍历Radio 
                if (obj[i].checked) {
                    item = obj[i].value;  
					break;
                }
		    

           }
            if(isfilemode)
			{
				item= "offline";
			}
		   console.log("asr mode"+item);
		   
		   return item;
}
		   

// 语音识别结果; 对jsonMsg数据解析,将识别结果附加到编辑框中
function getJsonMessage( jsonMsg ) {
	//console.log(jsonMsg);
	console.log( "message: " + JSON.parse(jsonMsg.data)['text'] );
	var rectxt=""+JSON.parse(jsonMsg.data)['text'];
	var asrmodel=JSON.parse(jsonMsg.data)['mode'];
	if(asrmodel=="2pass-offline")
	{
		offline_text=offline_text+rectxt; //.replace(/ +/g,"");
		rec_text=offline_text;
	}
	else
	{
		rec_text=rec_text+rectxt; //.replace(/ +/g,"");
	}
	var varArea=document.getElementById('varArea');
	
	varArea.value=rec_text;
	console.log( "offline_text: " + asrmodel+","+offline_text);
	console.log( "rec_text: " + rec_text);
	if (isfilemode==true){
		console.log("call stop ws!");
		play_file();
		wsconnecter.wsStop();
        
		info_div.innerHTML="请点击连接";
		isconnected=0;
		btnStart.disabled = true;
		btnStop.disabled = true;
		btnConnect.disabled=false;
	}
	
	 
 
}

// 连接状态响应
function getConnState( connState ) {
	if ( connState === 0 ) {
 
 
		info_div.innerHTML='连接成功!请点击开始';
		if (isfilemode==true){
			info_div.innerHTML='请耐心等待,大文件等待时间更长';
		}
	} else if ( connState === 1 ) {
		//stop();
	} else if ( connState === 2 ) {
		stop();
		console.log( 'connecttion error' );
		 
		alert("连接地址"+document.getElementById('wssip').value+"失败,请检查asr地址和端口，并确保h5服务和asr服务在同一个域内。或换个浏览器试试。");
		btnStart.disabled = true;
		isconnected=0;
 
		info_div.innerHTML='请点击连接';
	}
}

function record()
{
 
		 rec.open( function(){
		 rec.start();
		 console.log("开始");
		 btnStart.disabled = true;
		 });
 
}

 

// 识别启动、停止、清空操作
function start() {
	
	// 清除显示
	clear();
	//控件状态更新
 	console.log("isfilemode"+isfilemode+","+isconnected);
    info_div.innerHTML="正在连接asr服务器，请等待...";
	//启动连接
	var ret=wsconnecter.wsStart();
	if(ret==1){
		isRec = true;
		btnStart.disabled = false;
		btnStop.disabled = false;
		btnConnect.disabled=true;
		if (isfilemode)
		{
                 console.log("start file now");
			     start_file_offline();
 
				 btnStart.disabled = true;
		         btnStop.disabled = true;
		         btnConnect.disabled = true;
		}
        return 1;
	}
	return 0;
}

 
function stop() {
		var chunk_size = new Array( 5, 10, 5 );
		var request = {
			"chunk_size": chunk_size,
			"wav_name":  "h5",
			"is_speaking":  false,
			"chunk_interval":10,
			"mode":getAsrMode(),
		};
		console.log(request);
		if(sampleBuf.length>0){
		wsconnecter.wsSend(sampleBuf,false);
		console.log("sampleBuf.length"+sampleBuf.length);
		sampleBuf=new Int16Array();
		}
	   wsconnecter.wsSend( JSON.stringify(request) ,false);
 
	  
	
	 

	 //isconnected=0;
	// 控件状态更新
	
	isRec = false;
    info_div.innerHTML="发送完数据,请等候,正在识别...";

   if(isfilemode==false){
	    btnStop.disabled = true;
		btnStart.disabled = true;
		btnConnect.disabled=false;
	  setTimeout(function(){
		console.log("call stop ws!");
		wsconnecter.wsStop();
        isconnected=0;
		info_div.innerHTML="请点击连接";}, 3000 );
	   
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
   }
    // 停止连接
 
    

}

function clear() {
 
    var varArea=document.getElementById('varArea');
 
	varArea.value="";
    rec_text="";
	offline_text="";
 
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
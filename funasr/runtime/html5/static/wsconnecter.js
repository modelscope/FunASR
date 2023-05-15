/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
/* 2021-2023 by zhaoming,mali aihealthx.com */

function WebSocketConnectMethod( config ) { //定义socket连接方法类
	var Uri = "wss://111.205.137.58:5821/wss/" //设置wss asr online接口地址 如 wss://X.X.X.X:port/wss/
	var speechSokt;
	var connKeeperID;
	
	var msgHandle = config.msgHandle;
	var stateHandle = config.stateHandle;
			  
	this.wsStart = function () {
		
		if ( 'WebSocket' in window ) {
			speechSokt = new WebSocket( Uri ); // 定义socket连接对象
			speechSokt.onopen = function(e){onOpen(e);}; // 定义响应函数
			speechSokt.onclose = function(e){onClose(e);};
			speechSokt.onmessage = function(e){onMessage(e);};
			speechSokt.onerror = function(e){onError(e);};
		}
		else {
			alert('当前浏览器不支持 WebSocket');
		}
	};
	
	// 定义停止与发送函数
	this.wsStop = function () {
		if(speechSokt != undefined) {
			speechSokt.close();
		}
	};
	
	this.wsSend = function ( oneData,stop ) {
 
		if(speechSokt == undefined) return;
		if ( speechSokt.readyState === 1 ) { // 0:CONNECTING, 1:OPEN, 2:CLOSING, 3:CLOSED
 
			speechSokt.send( oneData );
			if(stop){
				setTimeout(speechSokt.close(), 3000 );
 
			}
			
		}
	};
	
	// SOCEKT连接中的消息与状态响应
	function onOpen( e ) {
		// 发送json
		var chunk_size = new Array( 5, 10, 5 );
		var request = {
			"chunk_size": chunk_size,
			"wav_name":  "h5",
			"is_speaking":  true,
			"chunk_interval":10,
		};
		speechSokt.send( JSON.stringify(request) );
		console.log("连接成功");
		stateHandle(0);
	}
	
	function onClose( e ) {
		stateHandle(1);
	}
	
	function onMessage( e ) {
 
		msgHandle( e );
	}
	
	function onError( e ) {
		console.log(e);
		stateHandle(2);
	}
    
 
}
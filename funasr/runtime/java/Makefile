
ENTRY_POINT = ./

 


WEBSOCKET_DIR:= ./
WEBSOCKET_FILES = \
	$(WEBSOCKET_DIR)/FunasrWsClient.java \
 


LIB_BUILD_DIR = ./lib


 

JAVAC = javac

BUILD_DIR = build


RUNJFLAGS = -Dfile.encoding=utf-8
	
	
vpath %.class $(BUILD_DIR)
vpath %.java src
	
	
 
	
rebuild: clean all
	
.PHONY:  clean run downjar

downjar:
	wget https://repo1.maven.org/maven2/org/slf4j/slf4j-api/1.7.25/slf4j-api-1.7.25.jar -P ./lib/
	wget https://repo1.maven.org/maven2/org/slf4j/slf4j-simple/1.7.25/slf4j-simple-1.7.25.jar -P ./lib/
	#wget https://github.com/TooTallNate/Java-WebSocket/releases/download/v1.5.3/Java-WebSocket-1.5.3.jar -P ./lib/
	wget https://repo1.maven.org/maven2/org/java-websocket/Java-WebSocket/1.5.3/Java-WebSocket-1.5.3.jar -P ./lib/
	wget https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/json-simple/json-simple-1.1.1.jar -P ./lib/
	wget https://github.com/argparse4j/argparse4j/releases/download/argparse4j-0.9.0/argparse4j-0.9.0.jar -P ./lib/
	rm -frv build
	mkdir build
clean:
	rm -frv  $(BUILD_DIR)/*
	rm -frv  $(LIB_BUILD_DIR)/*
	mkdir -p $(BUILD_DIR)
	mkdir -p ./lib

	
 

 

runclient:
		java   -cp $(BUILD_DIR):lib/Java-WebSocket-1.5.3.jar:lib/slf4j-simple-1.7.25.jar:lib/slf4j-api-1.7.25.jar:lib/json-simple-1.1.1.jar:lib/argparse4j-0.9.0.jar   $(RUNJFLAGS)    websocket.FunasrWsClient --host localhost --port 8889 --audio_in ./asr_example.wav --num_threads 1 --mode 2pass

 

buildwebsocket: $(WEBSOCKET_FILES:.java=.class)
	

%.class: %.java

				$(JAVAC) -cp $(BUILD_DIR):lib/slf4j-simple-1.7.25.jar:lib/slf4j-api-1.7.25.jar:lib/Java-WebSocket-1.5.3.jar:lib/json-simple-1.1.1.jar:lib/argparse4j-0.9.0.jar -d $(BUILD_DIR) -encoding UTF-8 $<

packjar:
				jar cvfe lib/funasrclient.jar . -C $(BUILD_DIR) .
				
all: clean buildlib packjar buildfile buildmic downjar buildwebsocket

 




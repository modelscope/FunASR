//
// Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
// Reserved. MIT License  (https://opensource.org/licenses/MIT)
//
/*
 * // 2022-2023 by zhaomingwork@qq.com
 */
// java FunasrWsClient
// usage:  FunasrWsClient [-h] [--port PORT] [--host HOST] [--audio_in AUDIO_IN] [--num_threads NUM_THREADS]
//                 [--chunk_size CHUNK_SIZE] [--chunk_interval CHUNK_INTERVAL] [--mode MODE]
package websocket;

import java.io.*;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.*;
import java.util.Map;
import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;
import org.java_websocket.client.WebSocketClient;
import org.java_websocket.drafts.Draft;
import org.java_websocket.handshake.ServerHandshake;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** This example demonstrates how to connect to websocket server. */
public class FunasrWsClient extends WebSocketClient {

  public class RecWavThread extends Thread {
    private FunasrWsClient funasrClient;

    public RecWavThread(FunasrWsClient funasrClient) {
      this.funasrClient = funasrClient;
    }

    public void run() {
      this.funasrClient.recWav();
    }
  }

  private static final Logger logger = LoggerFactory.getLogger(FunasrWsClient.class);

  public FunasrWsClient(URI serverUri, Draft draft) {
    super(serverUri, draft);
  }

  public FunasrWsClient(URI serverURI) {
    super(serverURI);
  }

  public FunasrWsClient(URI serverUri, Map<String, String> httpHeaders) {
    super(serverUri, httpHeaders);
  }

  public void getSslContext(String keyfile, String certfile) {
    // TODO
    return;
  }

  // send json at first time
  public void sendJson(
      String mode, String strChunkSize, int chunkInterval, String wavName, boolean isSpeaking) {
    try {

      JSONObject obj = new JSONObject();
      obj.put("mode", mode);
      JSONArray array = new JSONArray();
      String[] chunkList = strChunkSize.split(",");
      for (int i = 0; i < chunkList.length; i++) {
        array.add(Integer.valueOf(chunkList[i].trim()));
      }

      obj.put("chunk_size", array);
      obj.put("chunk_interval", new Integer(chunkInterval));
      obj.put("wav_name", wavName);
      if (isSpeaking) {
        obj.put("is_speaking", new Boolean(true));
      } else {
        obj.put("is_speaking", new Boolean(false));
      }
      logger.info("sendJson: " + obj);
      // return;

      send(obj.toString());

      return;
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  // send json at end of wav
  public void sendEof() {
    try {
      JSONObject obj = new JSONObject();

      obj.put("is_speaking", new Boolean(false));

      logger.info("sendEof: " + obj);
      // return;

      send(obj.toString());
      iseof = true;
      return;
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  // function for rec wav file
  public void recWav() {
    sendJson(mode, strChunkSize, chunkInterval, wavName, true);
    File file = new File(FunasrWsClient.wavPath);

    int chunkSize = sendChunkSize;
    byte[] bytes = new byte[chunkSize];

    int readSize = 0;
    try (FileInputStream fis = new FileInputStream(file)) {
      if (FunasrWsClient.wavPath.endsWith(".wav")) {
        fis.read(bytes, 0, 44); //skip first 44 wav header
      }
      readSize = fis.read(bytes, 0, chunkSize);
      while (readSize > 0) {
        // send when it is chunk size
        if (readSize == chunkSize) {
          send(bytes); // send buf to server

        } else {
          // send when at last or not is chunk size
          byte[] tmpBytes = new byte[readSize];
          for (int i = 0; i < readSize; i++) {
            tmpBytes[i] = bytes[i];
          }
          send(tmpBytes);
        }
        // if not in offline mode, we simulate online stream by sleep
        if (!mode.equals("offline")) {
          Thread.sleep(Integer.valueOf(chunkSize / 32));
        }

        readSize = fis.read(bytes, 0, chunkSize);
      }

      if (!mode.equals("offline")) {
        // if not offline, we send eof and wait for 3 seconds to close
        Thread.sleep(2000);
        sendEof();
        Thread.sleep(3000);
        close();
      } else {
        // if offline, just send eof
        sendEof();
      }

    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  @Override
  public void onOpen(ServerHandshake handshakedata) {

    RecWavThread thread = new RecWavThread(this);
    thread.start();
  }

  @Override
  public void onMessage(String message) {
    JSONObject jsonObject = new JSONObject();
    JSONParser jsonParser = new JSONParser();
    logger.info("received: " + message);
    try {
      jsonObject = (JSONObject) jsonParser.parse(message);
      logger.info("text: " + jsonObject.get("text"));
    } catch (org.json.simple.parser.ParseException e) {
      e.printStackTrace();
    }
    if (iseof && mode.equals("offline") && !jsonObject.containsKey("is_final")) {
      close();
    }
	 
    if (iseof && mode.equals("offline") && jsonObject.containsKey("is_final") && jsonObject.get("is_final").equals("false")) {
      close();
    }
  }

  @Override
  public void onClose(int code, String reason, boolean remote) {

    logger.info(
        "Connection closed by "
            + (remote ? "remote peer" : "us")
            + " Code: "
            + code
            + " Reason: "
            + reason);
  }

  @Override
  public void onError(Exception ex) {
    logger.info("ex: " + ex);
    ex.printStackTrace();
    // if the error is fatal then onClose will be called additionally
  }

  private boolean iseof = false;
  public static String wavPath;
  static String mode = "online";
  static String strChunkSize = "5,10,5";
  static int chunkInterval = 10;
  static int sendChunkSize = 1920;

  String wavName = "javatest";

  public static void main(String[] args) throws URISyntaxException {
    ArgumentParser parser = ArgumentParsers.newArgumentParser("ws client").defaultHelp(true);
    parser
        .addArgument("--port")
        .help("Port on which to listen.")
        .setDefault("8889")
        .type(String.class)
        .required(false);
    parser
        .addArgument("--host")
        .help("the IP address of server.")
        .setDefault("127.0.0.1")
        .type(String.class)
        .required(false);
    parser
        .addArgument("--audio_in")
        .help("wav path for decoding.")
        .setDefault("asr_example.wav")
        .type(String.class)
        .required(false);
    parser
        .addArgument("--num_threads")
        .help("num of threads for test.")
        .setDefault(1)
        .type(Integer.class)
        .required(false);
    parser
        .addArgument("--chunk_size")
        .help("chunk size for asr.")
        .setDefault("5, 10, 5")
        .type(String.class)
        .required(false);
    parser
        .addArgument("--chunk_interval")
        .help("chunk for asr.")
        .setDefault(10)
        .type(Integer.class)
        .required(false);

    parser
        .addArgument("--mode")
        .help("mode for asr.")
        .setDefault("offline")
        .type(String.class)
        .required(false);
    String srvIp = "";
    String srvPort = "";
    String wavPath = "";
    int numThreads = 1;
    String chunk_size = "";
    int chunk_interval = 10;
    String strmode = "offline";

    try {
      Namespace ns = parser.parseArgs(args);
      srvIp = ns.get("host");
      srvPort = ns.get("port");
      wavPath = ns.get("audio_in");
      numThreads = ns.get("num_threads");
      chunk_size = ns.get("chunk_size");
      chunk_interval = ns.get("chunk_interval");
      strmode = ns.get("mode");
      System.out.println(srvPort);

    } catch (ArgumentParserException ex) {
      ex.getParser().handleError(ex);
      return;
    }

    FunasrWsClient.strChunkSize = chunk_size;
    FunasrWsClient.chunkInterval = chunk_interval;
    FunasrWsClient.wavPath = wavPath;
    FunasrWsClient.mode = strmode;
    System.out.println(
        "serIp="
            + srvIp
            + ",srvPort="
            + srvPort
            + ",wavPath="
            + wavPath
            + ",strChunkSize"
            + strChunkSize);

    class ClientThread implements Runnable {

      String srvIp;
      String srvPort;

      ClientThread(String srvIp, String srvPort, String wavPath) {
        this.srvIp = srvIp;
        this.srvPort = srvPort;
      }

      public void run() {
        try {

          int RATE = 16000;
          String[] chunkList = strChunkSize.split(",");
          int int_chunk_size = 60 * Integer.valueOf(chunkList[1].trim()) / chunkInterval;
          int CHUNK = Integer.valueOf(RATE / 1000 * int_chunk_size);
          int stride =
              Integer.valueOf(
                  60 * Integer.valueOf(chunkList[1].trim()) / chunkInterval / 1000 * 16000 * 2);
          System.out.println("chunk_size:" + String.valueOf(int_chunk_size));
          System.out.println("CHUNK:" + CHUNK);
          System.out.println("stride:" + String.valueOf(stride));
          FunasrWsClient.sendChunkSize = CHUNK * 2;

          String wsAddress = "ws://" + srvIp + ":" + srvPort;

          FunasrWsClient c = new FunasrWsClient(new URI(wsAddress));

          c.connect();

          System.out.println("wsAddress:" + wsAddress);
        } catch (Exception e) {
          e.printStackTrace();
          System.out.println("e:" + e);
        }
      }
    }
    for (int i = 0; i < numThreads; i++) {
      System.out.println("Thread1 is running...");
      Thread t = new Thread(new ClientThread(srvIp, srvPort, wavPath));
      t.start();
    }
  }
}

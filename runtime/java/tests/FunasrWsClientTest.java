package websocket;

import java.net.URI;

public final class FunasrWsClientTest {
  private static final class RecordingClient extends FunasrWsClient {
    private boolean closeCalled;

    RecordingClient() throws Exception {
      super(new URI("ws://127.0.0.1:1"));
    }

    @Override
    public void close() {
      closeCalled = true;
    }

    @Override
    public void send(String text) {}
  }

  private static void assertCloseDecision(
      boolean expected, boolean eof, String clientMode, String message, String description)
      throws Exception {
    RecordingClient client = new RecordingClient();
    FunasrWsClient.mode = clientMode;
    if (eof) {
      client.sendEof();
    }

    client.onMessage(message);
    if (client.closeCalled != expected) {
      throw new AssertionError(
          description + ": expected close=" + expected + ", actual=" + client.closeCalled);
    }
  }

  public static void main(String[] args) throws Exception {
    assertCloseDecision(
        false,
        true,
        "offline",
        "{\"text\":\"partial\",\"is_final\":false}",
        "non-final offline response");
    assertCloseDecision(
        false,
        false,
        "offline",
        "{\"text\":\"done\",\"is_final\":true}",
        "final response before EOF");
    assertCloseDecision(
        false,
        true,
        "online",
        "{\"text\":\"done\",\"is_final\":true}",
        "final online response");
    assertCloseDecision(
        true,
        true,
        "offline",
        "{\"text\":\"legacy done\"}",
        "legacy offline response without is_final");
    assertCloseDecision(
        true,
        true,
        "offline",
        "{\"mode\":\"2pass-offline\",\"text\":\"done\",\"is_final\":true}",
        "final offline response");
  }
}

import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.net.InetAddress;
import java.net.Socket;
import java.util.Scanner;
import java.util.concurrent.TimeUnit;
import org.json.JSONObject;
public class TCPClientIntercept {

    private Socket socket;
    private Scanner scanner;
    private TCPClientIntercept(InetAddress serverAddress, int serverPort) throws Exception {
        this.socket = new Socket(serverAddress, serverPort);
        this.scanner = new Scanner(System.in);
    }
    private void start() throws IOException {
        //while (true) {
        try {
            TimeUnit.SECONDS.sleep(5);
        } catch (InterruptedException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        //int n = 1
        JSONObject jsonObject2 = new JSONObject();
        jsonObject2.put("1", 1);
        //jsonObject2.put("5", 1);

        OutputStream out = socket.getOutputStream();
        ObjectOutputStream o = new ObjectOutputStream(out);
        String stream = "01000000000000000000";
        o.writeObject(stream);
        
        sleep(2000);
        
        stream = "05000000000000000000";
        o.writeObject(stream);

        out.flush();
        o.close();
        System.out.println("Sent to server: " + " " + jsonObject2.get("key").toString());
        //}
    }

    public static void main(String[] args) throws Exception {
        TCPClientIntercept client = new TCPClientIntercept(
                InetAddress.getByName("192.168.43.220"),
                2001);

        System.out.println("\r\nConnected to Server: " + client.socket.getInetAddress());
        client.start();
    }
    
    public static void sleep(int time){
    	int i;
    	for(i = 0; i < time; i ++);
    }
} 
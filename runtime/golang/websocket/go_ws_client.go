package main

import (
	"bufio"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/go-audio/wav"
	"github.com/gorilla/websocket"
)

var args struct {
	audio_in       string
	thread_num     int
	host           string
	port           string
	output_dir     string
	hotword        string
	audio_fs       int
	use_itn        int
	mode           string
	chunk_size     []int
	chunk_interval int
}

var websocketConn *websocket.Conn
var offline_msg_done = false

type Message struct {
	WavName   string `json:"wav_name"`
	Text      string `json:"text"`
	TimeStamp string `json:"timestamp"`
	Mode      string `json:"mode"`
}

type AudioData struct {
	SampleRate int    `json:"sample_rate"`
	Stride     int    `json:"stride"`
	ChunkNum   int    `json:"chunk_num"`
	AudioBytes string `json:"audio_bytes"`
}

// 完全模拟Python wave库的行为读取WAV文件
func readWAVFile(filePath string, chunkSize []int, chunkInterval int) (*AudioData, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open WAV file: %v", err)
	}
	defer file.Close()

	// 使用wav库解码WAV文件，但只获取基本信息
	decoder := wav.NewDecoder(file)
	if !decoder.IsValidFile() {
		return nil, fmt.Errorf("invalid WAV file format")
	}

	// 获取WAV文件格式信息
	format := decoder.Format()
	sampleRate := int(format.SampleRate)

	// 读取所有音频数据
	buf, err := decoder.FullPCMBuffer()
	if err != nil {
		return nil, fmt.Errorf("failed to read full PCM data: %v", err)
	}

	// 获取原始音频字节数据（等同于Python的bytes(frames)）
	audioBytes := make([]byte, len(buf.Data)*2)
	for i, sample := range buf.Data {
		// 确保与Python wave库的字节序一致（小端序16位）
		sample16 := int16(sample)
		audioBytes[i*2] = byte(sample16 & 0xFF)
		audioBytes[i*2+1] = byte((sample16 >> 8) & 0xFF)
	}

	// 完全按照Python的计算方式计算参数
	// Python: stride = int(60 * chunk_size[1] / chunk_interval / 1000 * sample_rate * 2)
	// 注意：Python中的除法是浮点除法，Go中需要显式转换避免整数除法截断
	stride := int(float64(60*chunkSize[1]) / float64(chunkInterval) / 1000.0 * float64(sampleRate) * 2.0)

	// 添加安全检查防止除零
	if stride <= 0 {
		return nil, fmt.Errorf("calculated stride is zero or negative: %d", stride)
	}

	// Python: chunk_num = (len(audio_bytes) - 1) // stride + 1
	chunkNum := (len(audioBytes)-1)/stride + 1

	// 编码为Base64（与Python的base64.b64encode().decode('utf-8')一致）
	audioBase64 := base64.StdEncoding.EncodeToString(audioBytes)

	return &AudioData{
		SampleRate: sampleRate,
		Stride:     stride,
		ChunkNum:   chunkNum,
		AudioBytes: audioBase64,
	}, nil
}

func IntSlicetoString(nums []int) string {
	strNums := make([]string, len(nums))
	for i, num := range nums {
		strNums[i] = strconv.Itoa(num)
	}
	result := strings.Join(strNums, ",")
	return result
}

func recordFromScp(chunk_begin, chunk_size int) {
	wavs := []string{args.audio_in}
	sample_rate := args.audio_fs
	wav_format := "pcm"
	use_itn := true

	wav_name := "demo"
	wav_path := wavs[0]
	var audio_bytes []byte
	var stride, chunk_num int

	fst_dict := make(map[string]int)
	hotword_msg := ""
	if args.hotword != "" {
		file, err := os.Open(args.hotword)
		if err != nil {
			log.Fatalf("failed to open file: %v", err)
		}
		defer file.Close()

		scanner := bufio.NewScanner(file)
		for scanner.Scan() {
			words := strings.Fields(scanner.Text())

			if len(words) < 2 {
				fmt.Println("Please checkout format of hotwords")
				continue
			}
			weight, err := strconv.Atoi(words[1])
			if err != nil {
				fmt.Println("The weight of hotwords must be Int!")
			}
			fst_dict[words[0]] = weight
		}
		if err := scanner.Err(); err != nil {
			log.Fatalf("error reading file: %v", err)
		}
		bytes, _ := json.Marshal(fst_dict)
		hotword_msg = string(bytes)
		fmt.Println("HotWord: ", hotword_msg)
	}

	if args.use_itn == 0 {
		use_itn = false
	}

	if chunk_size > 0 {
		wavs = wavs[chunk_begin : chunk_begin+chunk_size]
	}

	if strings.HasSuffix(wav_path, ".wav") {
		// 使用Go原生实现替换Python调用
		audioData, err := readWAVFile(wav_path, args.chunk_size, args.chunk_interval)
		if err != nil {
			fmt.Println("Error reading WAV file:", err)
			return
		}

		stride = audioData.Stride
		chunk_num = audioData.ChunkNum
		sample_rate = audioData.SampleRate
		audio_bytes, err = base64.StdEncoding.DecodeString(audioData.AudioBytes)
		if err != nil {
			fmt.Println("Error decoding Base64:", err)
			return
		}
	} else {
		fmt.Println("Currently, only the WAV format is supported")
		return
	}

	first_message := make(map[string]interface{})
	first_message["mode"] = args.mode
	first_message["chunk_size"] = args.chunk_size
	first_message["chunk_interval"] = args.chunk_interval
	first_message["audio_fs"] = sample_rate
	first_message["wav_name"] = wav_name
	first_message["wav_format"] = wav_format
	first_message["is_speaking"] = true
	first_message["hotwords"] = hotword_msg
	first_message["itn"] = use_itn

	bytes, _ := json.Marshal(first_message)
	message := string(bytes)

	// fmt.Println(audio_bytes)
	// fmt.Println(stride)
	// fmt.Println(chunk_num)
	// fmt.Println(message)

	err := websocketConn.WriteMessage(websocket.TextMessage, []byte(message))
	if err != nil {
		log.Println("Failed to send the message:", err)
		return
	}

	is_speaking := true
	for i := 0; i < chunk_num; i++ {
		beg := i * stride
		var data []byte
		if i == chunk_num-1 {
			data = audio_bytes[beg:]
		} else {
			data = audio_bytes[beg : beg+stride]
		}

		err = websocketConn.WriteMessage(websocket.BinaryMessage, data)
		if err != nil {
			fmt.Println("Failed to send audio data:", err)
			return
		}

		if i == chunk_num-1 {
			is_speaking = false
			endMsg := map[string]bool{"is_speaking": is_speaking}
			endMsgBytes, err := json.Marshal(endMsg)
			if err != nil {
				fmt.Println("JSON serialization failed:", err)
				return
			}
			err = websocketConn.WriteMessage(websocket.TextMessage, endMsgBytes)
			if err != nil {
				fmt.Println("Failed to send the termination message:", err)
				return
			}
		}
		var sleepDuration time.Duration
		if args.mode == "offline" {
			sleepDuration = time.Millisecond
		} else {
			fmt.Println("timesleep:  Currently, only offline mode is supported.")
			// sleepDuration = time.Duration(60*float64(args.chunk_size[1])/float64(args.chunk_interval)) * time.Millisecond
			return
		}
		time.Sleep(sleepDuration)
	}

	if args.mode != "offline" {
		fmt.Println("Currently, only offline mode is supported.")
		return
	}
	if args.mode == "offline" {
		for !offline_msg_done {
			time.Sleep(1 * time.Second)
		}
	}
	websocketConn.Close()
}

func message(id string) {
	text_print := ""
	var ibestWriter *os.File
	var err error
	if args.output_dir != "" {
		filePath := fmt.Sprintf("%s/text.%s", args.output_dir, id)
		ibestWriter, err = os.OpenFile(filePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			log.Fatalf("failed to open file: %v", err)
		}
	} else {
		ibestWriter = nil
	}

	for {
		_, message, err := websocketConn.ReadMessage()
		if err != nil {
			log.Println("read:", err)
			break
		}
		var meg Message
		var wav_name string
		timestamp := ""

		err = json.Unmarshal(message, &meg)
		if err != nil {
			log.Println("unmarshal:", err)
			continue
		}

		if meg.WavName != "" {
			wav_name = meg.WavName
		} else {
			wav_name = "demo"
		}

		text := meg.Text

		if meg.TimeStamp != "" {
			timestamp = meg.TimeStamp
		}

		if ibestWriter != nil {
			var text_write_line string
			if timestamp != "" {
				text_write_line = fmt.Sprintf("%s\t%s\t%s\n", wav_name, text, timestamp)
			} else {
				text_write_line = fmt.Sprintf("%s\t%s\n", wav_name, text)
			}
			_, err = ibestWriter.WriteString(text_write_line)
			if err != nil {
				log.Fatalf("Failed to write to file: %v", err)
			}
		}

		if meg.Mode != "offline" {
			fmt.Println("Currently, only offline mode is supported.")
			return
		}

		if meg.Mode == "offline" {
			if timestamp != "" {
				text_print += fmt.Sprintf("%s timestamp: %s", text, timestamp)
			} else {
				text_print += fmt.Sprintf("%s ", text)
			}
			fmt.Println("\rpid" + id + ": " + wav_name + ": " + text_print)
			offline_msg_done = true
		}
	}
}

func wsClient(id, chunk_begin, chunk_size int, done chan bool) {
	for i := chunk_begin; i < chunk_begin+chunk_size; i++ {
		offline_msg_done = false

		u := url.URL{Scheme: "ws", Host: fmt.Sprintf("%s:%s", args.host, args.port), Path: "/"}
		fmt.Printf("Thread %d: Connecting to %s\n", id, u.String())
		var err error
		websocketConn, _, err = websocket.DefaultDialer.Dial(u.String(), nil)
		if err != nil {
			log.Fatal("dial:", err)
		}
		defer websocketConn.Close()

		var wg sync.WaitGroup
		wg.Add(2)

		go func() {
			defer wg.Done()
			recordFromScp(i, 1)
		}()

		go func() {
			defer wg.Done()
			id_str := strconv.Itoa(id)
			i_str := strconv.Itoa(i)
			message(id_str + "_" + i_str)
		}()

		wg.Wait()
	}

	done <- true
}

func oneThread(id, chunk_begin, chunk_size int, wg *sync.WaitGroup) {
	defer wg.Done()
	done := make(chan bool)

	go wsClient(id, chunk_begin, chunk_size, done)

	select {
	case <-done:
		fmt.Printf("Thread %d: Task completed\n", id)
	}
}

func main() {
	args.audio_in = "../audio/asr_example.wav"
	args.thread_num = 1
	args.host = "127.0.0.1"
	args.port = "10095"
	args.output_dir = "/workspace/models/Outputs"
	args.hotword = "/workspace/models/hotword.txt"
	args.chunk_size = []int{5, 10, 5}
	args.chunk_interval = 10
	args.mode = "offline"
	args.audio_fs = 16000
	args.use_itn = 1
	var chunk_size, remain_wavs int

	wavs := []string{args.audio_in}

	total_len := len(wavs)
	if total_len >= args.thread_num {
		chunk_size = total_len / args.thread_num
		remain_wavs = total_len - chunk_size*args.thread_num
	} else {
		chunk_size = 1
		remain_wavs = 0
	}

	var wg sync.WaitGroup
	chunk_begin := 0
	for i := 0; i < args.thread_num; i++ {
		wg.Add(1)
		now_chunk_size := chunk_size
		if remain_wavs > 0 {
			now_chunk_size = chunk_size + 1
			remain_wavs = remain_wavs - 1
		}
		go oneThread(i, chunk_begin, now_chunk_size, &wg)

		chunk_begin = chunk_begin + now_chunk_size
	}
	wg.Wait()
}

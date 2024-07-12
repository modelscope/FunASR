#!/usr/bin/env python3

import os
import argparse
import re

def contains_non_chinese(text):
    # 正则表达式匹配非中文字符
    non_chinese_pattern = re.compile(r'[^\u4e00-\u9fff]')
    return non_chinese_pattern.search(text) is not None

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--raw_data', type=str, help='data path')
    parser.add_argument('--outpath', type=str, help='output path')
    args = parser.parse_args()

    train_data_path=os.path.join(args.raw_data, "data_aishell2")
    dt_raw_data=os.path.join(args.raw_data, "devNtest")
    outpath=os.path.join(args.outpath,"data")

    os.makedirs(outpath,exist_ok=True)

    train_wav_path=os.path.join(train_data_path,"wav")
    train_txt_path=os.path.join(train_data_path,"transcript/trans.txt")

    device_list=["Android","IOS","MIC"]
    type_list=["dev","test"]

    train_wav_list=[]
    for root, dirs, files in os.walk(train_wav_path):
        for file in files:
            if file.endswith(".wav"):
                full_path = os.path.join(root, file)
                train_wav_list.append(full_path)
    print ("Done! Find the train wav length is",len(train_wav_list))

    id2wav = {}
    for wav in train_wav_list:
        id = wav.split("/")[-1].split(".")[0]
        id2wav[id] = wav
    print ("Done!")

    id2text = {}
    with open(train_txt_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]  # 清理每行的首尾空白字符
    for l in lines:
        id, txt = l.strip().split("	")
        id2text[id] = txt
    print ("Done! Find the train txt length is",len(lines))

    id2text_clean = {}
    id2wav_clean = {}
    for id, txt in id2text.items():
        if contains_non_chinese(txt):
            continue
        else:
            id2text_clean[id] = txt
            id2wav_clean[id] = id2wav[id]

    id2text = id2text_clean
    id2wav = id2wav_clean
    if len(id2text) != len(id2wav):
        print("Error: the number of wav files and text files is not equal!")
        exit(1)
    else:
        print ("Done! Finish clean the train data and the length is",len(id2text))

    train = os.path.join(outpath, "train")
    os.makedirs(train, exist_ok=True)

    trian_text = os.path.join(train, "text")
    trian_wav_scp = os.path.join(train, "wav.scp")

    with open(trian_text, 'w', encoding='utf-8') as file:
        for id, txt in id2text.items():
            file.write(id + " " + txt + "\n")

    with open(trian_wav_scp, 'w', encoding='utf-8') as file:
        for id, wav in id2wav.items():
            file.write(id + " " + wav + "\n")
        
    print ("Finish save the train data")

    for device in device_list:
        for type in type_list:
            print ("processing ", device,"-",type," data")
            data_path=os.path.join(dt_raw_data, device, type)
            wav_path=os.path.join(data_path,"wav")
            txt_path=os.path.join(data_path,"trans.txt")

            save_path=os.path.join(outpath, type, device)
            os.makedirs(save_path,exist_ok=True)
            text_path=os.path.join(save_path, "text")
            wav_scp_path=os.path.join(save_path, "wav.scp")

            wav_list=[]
            for root, dirs, files in os.walk(wav_path):
                for file in files:
                    if file.endswith(".wav"):
                        full_path = os.path.join(root, file)
                        wav_list.append(full_path)
            print ("Done! Find the ", device,"-",type, "wav length is",len(wav_list))

            id2wav = {}
            for wav in wav_list:
                id = wav.split("/")[-1].split(".")[0]
                id2wav[id] = wav

            id2text = {}
            with open(txt_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                lines = [line.strip() for line in lines]
            for l in lines:
                id, txt = l.strip().split("	")
                id2text[id] = txt

            id2text_clean = {}
            id2wav_clean = {}

            for id, txt in id2text.items():
                if contains_non_chinese(txt):
                    continue
                else:
                    id2text_clean[id] = txt
                    id2wav_clean[id] = id2wav[id]

            id2text = id2text_clean
            id2wav = id2wav_clean

            if len(id2text) != len(id2wav):
                print("Error: the number of wav files and text files is not equal!")
                exit(1)
            else:
                print ("Done! Find the ", device,"-",type, "txt length is",len(id2text))

            with open(text_path, 'w', encoding='utf-8') as file:
                for id, txt in id2text.items():
                    file.write(id + " " + txt + "\n")
            with open(wav_scp_path, 'w', encoding='utf-8') as file:
                for id, wav in id2wav.items():
                    file.write(id + " " + wav + "\n")

            print ("Finish save the ", device,"-",type, " data")

    print ("Finish all the data processing")




if __name__ == "__main__":
    main()
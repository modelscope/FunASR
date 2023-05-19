import sys
if __name__=="__main__":
    uttid_path=sys.argv[1]
    src_path=sys.argv[2]
    tgt_path=sys.argv[3]
    uttid_file=open(uttid_path,'r')
    uttid_line=uttid_file.readlines()
    uttid_file.close()
    ori_utt2spk_all_fifo_file=open(src_path+'/utt2spk_all_fifo','r')
    ori_utt2spk_all_fifo_line=ori_utt2spk_all_fifo_file.readlines()
    ori_utt2spk_all_fifo_file.close()
    new_utt2spk_all_fifo_file=open(tgt_path+'/utt2spk_all_fifo','w')

    uttid_list=[]
    for line in uttid_line:
        uttid_list.append(line.strip())
    
    for line in ori_utt2spk_all_fifo_line:
        if line.strip().split(' ')[0] in uttid_list:
            new_utt2spk_all_fifo_file.write(line)
    
    new_utt2spk_all_fifo_file.close()
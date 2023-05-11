import sys
if __name__=="__main__":
    path=sys.argv[1]

    text_id_old_file=open(path+"/text_id",'r')
    text_id_old=text_id_old_file.readlines()
    text_id_old_file.close()
    
    text_id=open(path+"/text_id_train",'w')
    for line in text_id_old:
        uttid=line.strip().split(' ')[0]
        old_id=line.strip().split(' ')[1]
        pre_id='0'
        new_id_list=[]
        for i in old_id:
            if i == '$':
                new_id_list.append(pre_id)
            else:
                new_id_list.append(str(int(i)-1))
                pre_id=str(int(i)-1)
        new_id_list.append(pre_id)
        new_id=' '.join(new_id_list)
        text_id.write(uttid+' '+new_id+'\n')
    text_id.close()
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_path', help='raw data path')
    args = parser.parse_args()

    root_path = args.root_path
    work_path = os.path.join(root_path, ".work")
    scp_files = os.listdir(work_path)

    reco2dur_dict = {}
    with open(root_path + 'reco2dur') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            reco2dur_dict[parts[0]] = parts[1]

    spk2utt_dict = {}
    with open(root_path + 'spk2utt') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            spk = parts[0]
            utts = parts[1:]
            for utt in utts:
                tmp = utt.split('data')
                rec = 'data_' + '_'.join(tmp[1][1:].split('_')[:-2])
                if rec in spk2utt_dict.keys():
                    spk2utt_dict[rec].append((spk, utt))
                else:
                    spk2utt_dict[rec] = []
                    spk2utt_dict[rec].append((spk, utt))

    segment_dict = {}
    with open(root_path + 'segments') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if parts[1] in segment_dict.keys():
                segment_dict[parts[1]].append((parts[0], parts[2], parts[3]))
            else:
                segment_dict[parts[1]] = []
                segment_dict[parts[1]].append((parts[0], parts[2], parts[3]))

    utt2spk_dict = {}
    with open(root_path + 'utt2spk') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            utt = parts[0]
            tmp = utt.split('data')
            rec = 'data_' + '_'.join(tmp[1][1:].split('_')[:-2])
            if rec in utt2spk_dict.keys():
                utt2spk_dict[rec].append((parts[0], parts[1]))
            else:
                utt2spk_dict[rec] = []
                utt2spk_dict[rec].append((parts[0], parts[1]))

    for file in scp_files:
        scp_file = work_path + file
        idx = scp_file.split('.')[-2]
        reco2dur_file = work_path + 'reco2dur.' + idx
        spk2utt_file = work_path + 'spk2utt.' + idx
        segment_file = work_path + 'segments.' + idx
        utt2spk_file = work_path + 'utt2spk.' + idx

        fpp = open(scp_file)
        scp_lines = fpp.readlines()
        keys = []
        for line in scp_lines:
            name = line.strip().split()[0]
            keys.append(name)

        with open(reco2dur_file, 'w') as f:
            lines = []
            for key in keys:
                string = key + ' ' + reco2dur_dict[key]
                lines.append(string + '\n')
            lines[-1] = lines[-1][:-1]
            f.writelines(lines)

        with open(spk2utt_file, 'w') as f:
            lines = []
            for key in keys:
                items = spk2utt_dict[key]
                for item in items:
                    string = item[0]
                    for it in item[1:]:
                        string += ' '
                        string += it
                    lines.append(string + '\n')
            lines[-1] = lines[-1][:-1]
            f.writelines(lines)

        with open(segment_file, 'w') as f:
            lines = []
            for key in keys:
                items = segment_dict[key]
                for item in items:
                    string = item[0] + ' ' + key + ' ' + item[1] + ' ' + item[2]
                    lines.append(string + '\n')
            lines[-1] = lines[-1][:-1]
            f.writelines(lines)

        with open(utt2spk_file, 'w') as f:
            lines = []
            for key in keys:
                items = utt2spk_dict[key]
                for item in items:
                    string = item[0] + ' ' + item[1]
                    lines.append(string + '\n')
            lines[-1] = lines[-1][:-1]
            f.writelines(lines)

        fpp.close()

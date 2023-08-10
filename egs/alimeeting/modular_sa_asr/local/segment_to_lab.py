import argparse
import os


def read_segments_file(segments_file):
    utt2segments = dict()
    with open(segments_file, "r") as fr:
        lines = fr.readlines()
        for line in lines:
            parts = line.strip().split()
            segment_utt_id, utt_id, start, end = parts[0], parts[1], float(parts[2]), float(parts[3])
            if utt_id not in utt2segments:
                utt2segments[utt_id] = []
            utt2segments[utt_id].append((segment_utt_id, start, end))
    return utt2segments


def write_label(label_file, label_list):
    with open(label_file, "w") as fw:
        for (start, end) in label_list:
            fw.write(f"{start} {end} sp\n")
        fw.flush()


def write_label_scp_file(label_scp_file, label_scp: dict):
    with open(label_scp_file, "w") as fw:
        for (utt_id, label_path) in label_scp.items():
            fw.write(f"{utt_id} {label_path}\n")
        fw.flush()


def main(args):
    input_segments = args.input_segments
    label_path = args.label_path
    output_label_scp_file = args.output_label_scp_file

    utt2segments = read_segments_file(input_segments)
    print(f"Collect {len(utt2segments)} utt2segments in file {input_segments}")

    result_label_scp = dict()
    for utt_id in utt2segments.keys():
        segment_list = utt2segments[utt_id]
        cur_label_path = os.path.join(label_path, f"{utt_id}.lab")
        write_label(cur_label_path, label_list=[(i1, i2) for (_, i1, i2) in segment_list])
        result_label_scp[utt_id] = cur_label_path
    write_label_scp_file(output_label_scp_file, result_label_scp)
    print(f"Write {len(result_label_scp)} labels")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make the lab file for segments")
    parser.add_argument("--input_segments", required=True, help="The input segments file")
    parser.add_argument("--label_path", required=True, help="The label_path to save file.lab")
    parser.add_argument("--output_label_scp_file", required=True, help="The output label.scp file")

    args = parser.parse_args()
    main(args)


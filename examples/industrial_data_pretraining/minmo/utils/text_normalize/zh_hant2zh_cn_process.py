import zhconv
import argparse
import codecs


def convert_hant2cn(input_file, output_file):
    fout = codecs.open(output_file, "w")
    with codecs.open(input_file, "r") as fin:
        for line in fin:
            if "\t" in line:
                content = line.strip().split("\t", 1)
            else:
                content = line.strip().split(" ", 1)
            if len(content) == 2:
                idx, res = content[0], content[1]
            else:
                idx = content[0]
                res = ""
            convert_res = zhconv.convert(res, "zh-cn")
            # print(idx, res, convert_res)
            fout.writelines(idx + "\t" + convert_res + "\n")

    fout.close()


parser = argparse.ArgumentParser(description="manual to this script")
parser.add_argument("--input_file", type=str, default=None)
parser.add_argument("--output_file", type=str, default=None)
args = parser.parse_args()

if __name__ == "__main__":
    convert_hant2cn(args.input_file, args.output_file)

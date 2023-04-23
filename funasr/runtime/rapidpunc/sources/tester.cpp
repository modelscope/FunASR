#include "precomp.h"
#define YML_FILE "/root/.vs/qmpunc/c29b6b8c-ce62-4d35-a660-861ded8b7c0e/src/models/punc.yaml"
#define TXT_FILE "/root/.vs/qmpunc/c29b6b8c-ce62-4d35-a660-861ded8b7c0e/src/testfiles/funasr.txt"

#define MODEL_DIR "/opt/models/punc/"
#include <fstream>


int main(int argc, char** argv)
{
		
	//auto Tokenizer = CRpTokenizer(YML_FILE);

	//auto id = Tokenizer.String2ID("§°f");

	ifstream ifile(TXT_FILE);
	if (!ifile.is_open())
		return -1;


	ostringstream is;
	is << ifile.rdbuf();


	auto Handle = RapidPuncInit(MODEL_DIR, RPPUNC_DEFAULT_THREADNUM);
	if (!Handle)
	{
		cout << "Bad Initalization" << endl;
		return -1;
	}
	auto Result = RapidPuncAddPunc(Handle, is.str().c_str());
	if (Result)
	{
		cout << RapidPuncGetResultText(Result) << endl;
		RapidPuncFree(Result);
	}

	RapidPuncFinal(Handle);


	return 0;
}




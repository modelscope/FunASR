#pragma once 
// system 
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <deque>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <list>
#include <locale.h>
#include <vector>
#include <string>
#include <math.h>
#include <numeric>
#include <cstring>

using namespace std;
// third part
#include "onnxruntime_run_options_config_keys.h"
#include "onnxruntime_cxx_api.h"
#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/online-feature.h"

// mine
#include "ComDefine.h"
#include "commonfunc.h"
#include "predefine_coe.h"
#include "tokenizer.h"
#include "CT-transformer.h"
#include "FsmnVad.h"
#include "e2e_vad.h"
#include "Vocab.h"
#include "CommonStruct.h"
#include "Audio.h"
#include "Tensor.h"
#include "util.h"
#include "resample.h"
#include "Model.h"
#include "paraformer_onnx.h"
#include "libfunasrapi.h"

using namespace paraformer;

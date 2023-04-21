#pragma once


// system
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <map>
#include <iterator>
#include <regex>
#include <algorithm>
#include <numeric>
using namespace std;


// third-part

#include <yaml-cpp/yaml.h>



#include "onnxruntime/onnxruntime_run_options_config_keys.h"
#include "onnxruntime/onnxruntime_cxx_api.h"




// ours
#include "constdef.h"

#include "commonfunc.h"

#include "tokenizer.h"

#include "punc_infer.h"

#include "libpuncapi.h"
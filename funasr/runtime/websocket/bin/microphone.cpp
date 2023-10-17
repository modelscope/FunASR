/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */

#include "microphone.h"

#include <stdio.h>
#include <stdlib.h>

#include "portaudio.h"  // NOLINT

Microphone::Microphone() {
  PaError err = Pa_Initialize();
  if (err != paNoError) {
    LOG(ERROR)<<"portaudio error: " << Pa_GetErrorText(err);
    exit(-1);
  }
}

Microphone::~Microphone() {
  PaError err = Pa_Terminate();
  if (err != paNoError) {
    LOG(ERROR)<<"portaudio error: " << Pa_GetErrorText(err);
    exit(-1);
  }
}

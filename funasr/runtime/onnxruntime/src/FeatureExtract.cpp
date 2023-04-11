
#include "precomp.h"

using namespace std;

FeatureExtract::FeatureExtract(int mode) : mode(mode)
{
}

FeatureExtract::~FeatureExtract()
{
}

void FeatureExtract::reset()
{
    speech.reset();
    fqueue.reset();
}

int FeatureExtract::size()
{
    return fqueue.size();
}

void FeatureExtract::insert(fftwf_plan plan, float *din, int len, int flag)
{
    float* fft_input = (float *)fftwf_malloc(sizeof(float) * fft_size);
    fftwf_complex* fft_out = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * fft_size);
    memset(fft_input, 0, sizeof(float) * fft_size);

    const float *window = (const float *)&window_hex;
    if (mode == 3)
        window = (const float *)&window_hamm_hex;

    speech.load(din, len);
    int i, j;
    float tmp_feature[80];
    if (mode == 0 || mode == 2 || mode == 3) {
        int ll = (speech.size() - window_size) / window_shift + 1;
        fqueue.reinit(ll);
    }

    for (i = 0; i <= speech.size() - window_size; i = i + window_shift) {
        float tmp_mean = 0;
        for (j = 0; j < window_size; j++) {
            tmp_mean += speech[i + j];
        }

        tmp_mean = tmp_mean / window_size;

        float pre_val = (float)speech[i] - tmp_mean;

        for (j = 0; j < window_size; j++) {
            float win = window[j];
            float cur_val = (float)speech[i + j] - tmp_mean;
            fft_input[j] = win * (cur_val - 0.97 * pre_val);
            pre_val = cur_val;
        }

        fftwf_execute_dft_r2c(plan, fft_input, fft_out);

        melspect((float *)fft_out, tmp_feature);
        int tmp_flag = S_MIDDLE;
        if (flag == S_END && i > speech.size() - 560)
            tmp_flag = S_END;

        fqueue.push(tmp_feature, tmp_flag);
    }
    speech.update(i);
    fftwf_free(fft_input);
    fftwf_free(fft_out);
}

bool FeatureExtract::fetch(Tensor<float> *&dout)
{
    if (fqueue.size() < 1) {
        return false;
    } else {
        dout = fqueue.pop();
        return true;
    }
}

void FeatureExtract::global_cmvn(float *din)
{
    const float *std;
    const float *mean;

    if (mode < 2) {
        if (mode == 0) {
            std = (const float *)global_cmvn_std_hex;
            mean = (const float *)global_cmvn_mean_hex;
        } else {
            std = (const float *)global_cmvn_std_online_hex;
            mean = (const float *)global_cmvn_mean_online_hex;
        }

        int i;
        for (i = 0; i < 80; i++) {
            float tmp = din[i] < 1e-7 ? 1e-7 : din[i];
            tmp = log(tmp);
            din[i] = (tmp - mean[i]) / std[i];
        }
    } else {
        int i;

        int val = 0x34000000;
        float min_resol = *((float *)&val);

        for (i = 0; i < 80; i++) {
            float tmp = din[i] < min_resol ? min_resol : din[i];
            din[i] = log(tmp);
        }
    }
}

void FeatureExtract::melspect(float *din, float *dout)
{
    float fftmag[256];
    const float *melcoe = (const float *)melcoe_hex;
    int i;
    for (i = 0; i < 256; i++) {
        float real = din[2 * i];
        float imag = din[2 * i + 1];
        fftmag[i] = real * real + imag * imag;
    }
    dout[0] = melcoe[0] * fftmag[1] + melcoe[1] * fftmag[2];
    dout[1] = melcoe[2] * fftmag[2];
    dout[2] = melcoe[3] * fftmag[3];
    dout[3] = melcoe[4] * fftmag[3] + melcoe[5] * fftmag[4];
    dout[4] = melcoe[6] * fftmag[4] + melcoe[7] * fftmag[5];
    dout[5] = melcoe[8] * fftmag[5] + melcoe[9] * fftmag[6];
    dout[6] = melcoe[10] * fftmag[6] + melcoe[11] * fftmag[7];
    dout[7] = melcoe[12] * fftmag[7];
    dout[8] = melcoe[13] * fftmag[8];
    dout[9] = melcoe[14] * fftmag[8] + melcoe[15] * fftmag[9];
    dout[10] = melcoe[16] * fftmag[9] + melcoe[17] * fftmag[10];
    dout[11] = melcoe[18] * fftmag[10] + melcoe[19] * fftmag[11];
    dout[12] = melcoe[20] * fftmag[11] + melcoe[21] * fftmag[12] +
               melcoe[22] * fftmag[13];
    dout[13] = melcoe[23] * fftmag[12] + melcoe[24] * fftmag[13] +
               melcoe[25] * fftmag[14];
    dout[14] = melcoe[26] * fftmag[14] + melcoe[27] * fftmag[15];
    dout[15] = melcoe[28] * fftmag[15] + melcoe[29] * fftmag[16];
    dout[16] = melcoe[30] * fftmag[16] + melcoe[31] * fftmag[17];
    dout[17] = melcoe[32] * fftmag[17] + melcoe[33] * fftmag[18];
    dout[18] = melcoe[34] * fftmag[18] + melcoe[35] * fftmag[19] +
               melcoe[36] * fftmag[20];
    dout[19] = melcoe[37] * fftmag[19] + melcoe[38] * fftmag[20] +
               melcoe[39] * fftmag[21];
    dout[20] = melcoe[40] * fftmag[21] + melcoe[41] * fftmag[22];
    dout[21] = melcoe[42] * fftmag[22] + melcoe[43] * fftmag[23] +
               melcoe[44] * fftmag[24];
    dout[22] = melcoe[45] * fftmag[23] + melcoe[46] * fftmag[24] +
               melcoe[47] * fftmag[25];
    dout[23] = melcoe[48] * fftmag[25] + melcoe[49] * fftmag[26] +
               melcoe[50] * fftmag[27];
    dout[24] = melcoe[51] * fftmag[26] + melcoe[52] * fftmag[27] +
               melcoe[53] * fftmag[28];
    dout[25] = melcoe[54] * fftmag[28] + melcoe[55] * fftmag[29] +
               melcoe[56] * fftmag[30];
    dout[26] = melcoe[57] * fftmag[29] + melcoe[58] * fftmag[30] +
               melcoe[59] * fftmag[31] + melcoe[60] * fftmag[32];
    dout[27] = melcoe[61] * fftmag[31] + melcoe[62] * fftmag[32] +
               melcoe[63] * fftmag[33];
    dout[28] = melcoe[64] * fftmag[33] + melcoe[65] * fftmag[34] +
               melcoe[66] * fftmag[35];
    dout[29] = melcoe[67] * fftmag[34] + melcoe[68] * fftmag[35] +
               melcoe[69] * fftmag[36] + melcoe[70] * fftmag[37];
    dout[30] = melcoe[71] * fftmag[36] + melcoe[72] * fftmag[37] +
               melcoe[73] * fftmag[38] + melcoe[74] * fftmag[39];
    dout[31] = melcoe[75] * fftmag[38] + melcoe[76] * fftmag[39] +
               melcoe[77] * fftmag[40] + melcoe[78] * fftmag[41];
    dout[32] = melcoe[79] * fftmag[40] + melcoe[80] * fftmag[41] +
               melcoe[81] * fftmag[42] + melcoe[82] * fftmag[43];
    dout[33] = melcoe[83] * fftmag[42] + melcoe[84] * fftmag[43] +
               melcoe[85] * fftmag[44] + melcoe[86] * fftmag[45];
    dout[34] = melcoe[87] * fftmag[44] + melcoe[88] * fftmag[45] +
               melcoe[89] * fftmag[46] + melcoe[90] * fftmag[47];
    dout[35] = melcoe[91] * fftmag[46] + melcoe[92] * fftmag[47] +
               melcoe[93] * fftmag[48] + melcoe[94] * fftmag[49];
    dout[36] = melcoe[95] * fftmag[48] + melcoe[96] * fftmag[49] +
               melcoe[97] * fftmag[50] + melcoe[98] * fftmag[51];
    dout[37] = melcoe[99] * fftmag[50] + melcoe[100] * fftmag[51] +
               melcoe[101] * fftmag[52] + melcoe[102] * fftmag[53] +
               melcoe[103] * fftmag[54];
    dout[38] = melcoe[104] * fftmag[52] + melcoe[105] * fftmag[53] +
               melcoe[106] * fftmag[54] + melcoe[107] * fftmag[55] +
               melcoe[108] * fftmag[56];
    dout[39] = melcoe[109] * fftmag[55] + melcoe[110] * fftmag[56] +
               melcoe[111] * fftmag[57] + melcoe[112] * fftmag[58];
    dout[40] = melcoe[113] * fftmag[57] + melcoe[114] * fftmag[58] +
               melcoe[115] * fftmag[59] + melcoe[116] * fftmag[60] +
               melcoe[117] * fftmag[61];
    dout[41] = melcoe[118] * fftmag[59] + melcoe[119] * fftmag[60] +
               melcoe[120] * fftmag[61] + melcoe[121] * fftmag[62] +
               melcoe[122] * fftmag[63] + melcoe[123] * fftmag[64];
    dout[42] = melcoe[124] * fftmag[62] + melcoe[125] * fftmag[63] +
               melcoe[126] * fftmag[64] + melcoe[127] * fftmag[65] +
               melcoe[128] * fftmag[66];
    dout[43] = melcoe[129] * fftmag[65] + melcoe[130] * fftmag[66] +
               melcoe[131] * fftmag[67] + melcoe[132] * fftmag[68] +
               melcoe[133] * fftmag[69];
    dout[44] = melcoe[134] * fftmag[67] + melcoe[135] * fftmag[68] +
               melcoe[136] * fftmag[69] + melcoe[137] * fftmag[70] +
               melcoe[138] * fftmag[71] + melcoe[139] * fftmag[72];
    dout[45] = melcoe[140] * fftmag[70] + melcoe[141] * fftmag[71] +
               melcoe[142] * fftmag[72] + melcoe[143] * fftmag[73] +
               melcoe[144] * fftmag[74] + melcoe[145] * fftmag[75];
    dout[46] = melcoe[146] * fftmag[73] + melcoe[147] * fftmag[74] +
               melcoe[148] * fftmag[75] + melcoe[149] * fftmag[76] +
               melcoe[150] * fftmag[77] + melcoe[151] * fftmag[78];
    dout[47] = melcoe[152] * fftmag[76] + melcoe[153] * fftmag[77] +
               melcoe[154] * fftmag[78] + melcoe[155] * fftmag[79] +
               melcoe[156] * fftmag[80] + melcoe[157] * fftmag[81];
    dout[48] = melcoe[158] * fftmag[79] + melcoe[159] * fftmag[80] +
               melcoe[160] * fftmag[81] + melcoe[161] * fftmag[82] +
               melcoe[162] * fftmag[83] + melcoe[163] * fftmag[84];
    dout[49] = melcoe[164] * fftmag[82] + melcoe[165] * fftmag[83] +
               melcoe[166] * fftmag[84] + melcoe[167] * fftmag[85] +
               melcoe[168] * fftmag[86] + melcoe[169] * fftmag[87] +
               melcoe[170] * fftmag[88];
    dout[50] = melcoe[171] * fftmag[85] + melcoe[172] * fftmag[86] +
               melcoe[173] * fftmag[87] + melcoe[174] * fftmag[88] +
               melcoe[175] * fftmag[89] + melcoe[176] * fftmag[90] +
               melcoe[177] * fftmag[91];
    dout[51] = melcoe[178] * fftmag[89] + melcoe[179] * fftmag[90] +
               melcoe[180] * fftmag[91] + melcoe[181] * fftmag[92] +
               melcoe[182] * fftmag[93] + melcoe[183] * fftmag[94] +
               melcoe[184] * fftmag[95];
    dout[52] = melcoe[185] * fftmag[92] + melcoe[186] * fftmag[93] +
               melcoe[187] * fftmag[94] + melcoe[188] * fftmag[95] +
               melcoe[189] * fftmag[96] + melcoe[190] * fftmag[97] +
               melcoe[191] * fftmag[98];
    dout[53] = melcoe[192] * fftmag[96] + melcoe[193] * fftmag[97] +
               melcoe[194] * fftmag[98] + melcoe[195] * fftmag[99] +
               melcoe[196] * fftmag[100] + melcoe[197] * fftmag[101] +
               melcoe[198] * fftmag[102];
    dout[54] = melcoe[199] * fftmag[99] + melcoe[200] * fftmag[100] +
               melcoe[201] * fftmag[101] + melcoe[202] * fftmag[102] +
               melcoe[203] * fftmag[103] + melcoe[204] * fftmag[104] +
               melcoe[205] * fftmag[105] + melcoe[206] * fftmag[106];
    dout[55] = melcoe[207] * fftmag[103] + melcoe[208] * fftmag[104] +
               melcoe[209] * fftmag[105] + melcoe[210] * fftmag[106] +
               melcoe[211] * fftmag[107] + melcoe[212] * fftmag[108] +
               melcoe[213] * fftmag[109] + melcoe[214] * fftmag[110];
    dout[56] = melcoe[215] * fftmag[107] + melcoe[216] * fftmag[108] +
               melcoe[217] * fftmag[109] + melcoe[218] * fftmag[110] +
               melcoe[219] * fftmag[111] + melcoe[220] * fftmag[112] +
               melcoe[221] * fftmag[113] + melcoe[222] * fftmag[114];
    dout[57] = melcoe[223] * fftmag[111] + melcoe[224] * fftmag[112] +
               melcoe[225] * fftmag[113] + melcoe[226] * fftmag[114] +
               melcoe[227] * fftmag[115] + melcoe[228] * fftmag[116] +
               melcoe[229] * fftmag[117] + melcoe[230] * fftmag[118] +
               melcoe[231] * fftmag[119];
    dout[58] = melcoe[232] * fftmag[115] + melcoe[233] * fftmag[116] +
               melcoe[234] * fftmag[117] + melcoe[235] * fftmag[118] +
               melcoe[236] * fftmag[119] + melcoe[237] * fftmag[120] +
               melcoe[238] * fftmag[121] + melcoe[239] * fftmag[122] +
               melcoe[240] * fftmag[123];
    dout[59] = melcoe[241] * fftmag[120] + melcoe[242] * fftmag[121] +
               melcoe[243] * fftmag[122] + melcoe[244] * fftmag[123] +
               melcoe[245] * fftmag[124] + melcoe[246] * fftmag[125] +
               melcoe[247] * fftmag[126] + melcoe[248] * fftmag[127] +
               melcoe[249] * fftmag[128];
    dout[60] = melcoe[250] * fftmag[124] + melcoe[251] * fftmag[125] +
               melcoe[252] * fftmag[126] + melcoe[253] * fftmag[127] +
               melcoe[254] * fftmag[128] + melcoe[255] * fftmag[129] +
               melcoe[256] * fftmag[130] + melcoe[257] * fftmag[131] +
               melcoe[258] * fftmag[132];
    dout[61] = melcoe[259] * fftmag[129] + melcoe[260] * fftmag[130] +
               melcoe[261] * fftmag[131] + melcoe[262] * fftmag[132] +
               melcoe[263] * fftmag[133] + melcoe[264] * fftmag[134] +
               melcoe[265] * fftmag[135] + melcoe[266] * fftmag[136] +
               melcoe[267] * fftmag[137];
    dout[62] = melcoe[268] * fftmag[133] + melcoe[269] * fftmag[134] +
               melcoe[270] * fftmag[135] + melcoe[271] * fftmag[136] +
               melcoe[272] * fftmag[137] + melcoe[273] * fftmag[138] +
               melcoe[274] * fftmag[139] + melcoe[275] * fftmag[140] +
               melcoe[276] * fftmag[141] + melcoe[277] * fftmag[142];
    dout[63] = melcoe[278] * fftmag[138] + melcoe[279] * fftmag[139] +
               melcoe[280] * fftmag[140] + melcoe[281] * fftmag[141] +
               melcoe[282] * fftmag[142] + melcoe[283] * fftmag[143] +
               melcoe[284] * fftmag[144] + melcoe[285] * fftmag[145] +
               melcoe[286] * fftmag[146] + melcoe[287] * fftmag[147];
    dout[64] = melcoe[288] * fftmag[143] + melcoe[289] * fftmag[144] +
               melcoe[290] * fftmag[145] + melcoe[291] * fftmag[146] +
               melcoe[292] * fftmag[147] + melcoe[293] * fftmag[148] +
               melcoe[294] * fftmag[149] + melcoe[295] * fftmag[150] +
               melcoe[296] * fftmag[151] + melcoe[297] * fftmag[152] +
               melcoe[298] * fftmag[153];
    dout[65] = melcoe[299] * fftmag[148] + melcoe[300] * fftmag[149] +
               melcoe[301] * fftmag[150] + melcoe[302] * fftmag[151] +
               melcoe[303] * fftmag[152] + melcoe[304] * fftmag[153] +
               melcoe[305] * fftmag[154] + melcoe[306] * fftmag[155] +
               melcoe[307] * fftmag[156] + melcoe[308] * fftmag[157] +
               melcoe[309] * fftmag[158];
    dout[66] = melcoe[310] * fftmag[154] + melcoe[311] * fftmag[155] +
               melcoe[312] * fftmag[156] + melcoe[313] * fftmag[157] +
               melcoe[314] * fftmag[158] + melcoe[315] * fftmag[159] +
               melcoe[316] * fftmag[160] + melcoe[317] * fftmag[161] +
               melcoe[318] * fftmag[162] + melcoe[319] * fftmag[163] +
               melcoe[320] * fftmag[164];
    dout[67] = melcoe[321] * fftmag[159] + melcoe[322] * fftmag[160] +
               melcoe[323] * fftmag[161] + melcoe[324] * fftmag[162] +
               melcoe[325] * fftmag[163] + melcoe[326] * fftmag[164] +
               melcoe[327] * fftmag[165] + melcoe[328] * fftmag[166] +
               melcoe[329] * fftmag[167] + melcoe[330] * fftmag[168] +
               melcoe[331] * fftmag[169] + melcoe[332] * fftmag[170];
    dout[68] = melcoe[333] * fftmag[165] + melcoe[334] * fftmag[166] +
               melcoe[335] * fftmag[167] + melcoe[336] * fftmag[168] +
               melcoe[337] * fftmag[169] + melcoe[338] * fftmag[170] +
               melcoe[339] * fftmag[171] + melcoe[340] * fftmag[172] +
               melcoe[341] * fftmag[173] + melcoe[342] * fftmag[174] +
               melcoe[343] * fftmag[175] + melcoe[344] * fftmag[176];
    dout[69] = melcoe[345] * fftmag[171] + melcoe[346] * fftmag[172] +
               melcoe[347] * fftmag[173] + melcoe[348] * fftmag[174] +
               melcoe[349] * fftmag[175] + melcoe[350] * fftmag[176] +
               melcoe[351] * fftmag[177] + melcoe[352] * fftmag[178] +
               melcoe[353] * fftmag[179] + melcoe[354] * fftmag[180] +
               melcoe[355] * fftmag[181] + melcoe[356] * fftmag[182];
    dout[70] = melcoe[357] * fftmag[177] + melcoe[358] * fftmag[178] +
               melcoe[359] * fftmag[179] + melcoe[360] * fftmag[180] +
               melcoe[361] * fftmag[181] + melcoe[362] * fftmag[182] +
               melcoe[363] * fftmag[183] + melcoe[364] * fftmag[184] +
               melcoe[365] * fftmag[185] + melcoe[366] * fftmag[186] +
               melcoe[367] * fftmag[187] + melcoe[368] * fftmag[188];
    dout[71] = melcoe[369] * fftmag[183] + melcoe[370] * fftmag[184] +
               melcoe[371] * fftmag[185] + melcoe[372] * fftmag[186] +
               melcoe[373] * fftmag[187] + melcoe[374] * fftmag[188] +
               melcoe[375] * fftmag[189] + melcoe[376] * fftmag[190] +
               melcoe[377] * fftmag[191] + melcoe[378] * fftmag[192] +
               melcoe[379] * fftmag[193] + melcoe[380] * fftmag[194] +
               melcoe[381] * fftmag[195];
    dout[72] = melcoe[382] * fftmag[189] + melcoe[383] * fftmag[190] +
               melcoe[384] * fftmag[191] + melcoe[385] * fftmag[192] +
               melcoe[386] * fftmag[193] + melcoe[387] * fftmag[194] +
               melcoe[388] * fftmag[195] + melcoe[389] * fftmag[196] +
               melcoe[390] * fftmag[197] + melcoe[391] * fftmag[198] +
               melcoe[392] * fftmag[199] + melcoe[393] * fftmag[200] +
               melcoe[394] * fftmag[201] + melcoe[395] * fftmag[202];
    dout[73] = melcoe[396] * fftmag[196] + melcoe[397] * fftmag[197] +
               melcoe[398] * fftmag[198] + melcoe[399] * fftmag[199] +
               melcoe[400] * fftmag[200] + melcoe[401] * fftmag[201] +
               melcoe[402] * fftmag[202] + melcoe[403] * fftmag[203] +
               melcoe[404] * fftmag[204] + melcoe[405] * fftmag[205] +
               melcoe[406] * fftmag[206] + melcoe[407] * fftmag[207] +
               melcoe[408] * fftmag[208] + melcoe[409] * fftmag[209];
    dout[74] = melcoe[410] * fftmag[203] + melcoe[411] * fftmag[204] +
               melcoe[412] * fftmag[205] + melcoe[413] * fftmag[206] +
               melcoe[414] * fftmag[207] + melcoe[415] * fftmag[208] +
               melcoe[416] * fftmag[209] + melcoe[417] * fftmag[210] +
               melcoe[418] * fftmag[211] + melcoe[419] * fftmag[212] +
               melcoe[420] * fftmag[213] + melcoe[421] * fftmag[214] +
               melcoe[422] * fftmag[215] + melcoe[423] * fftmag[216];
    dout[75] = melcoe[424] * fftmag[210] + melcoe[425] * fftmag[211] +
               melcoe[426] * fftmag[212] + melcoe[427] * fftmag[213] +
               melcoe[428] * fftmag[214] + melcoe[429] * fftmag[215] +
               melcoe[430] * fftmag[216] + melcoe[431] * fftmag[217] +
               melcoe[432] * fftmag[218] + melcoe[433] * fftmag[219] +
               melcoe[434] * fftmag[220] + melcoe[435] * fftmag[221] +
               melcoe[436] * fftmag[222] + melcoe[437] * fftmag[223];
    dout[76] = melcoe[438] * fftmag[217] + melcoe[439] * fftmag[218] +
               melcoe[440] * fftmag[219] + melcoe[441] * fftmag[220] +
               melcoe[442] * fftmag[221] + melcoe[443] * fftmag[222] +
               melcoe[444] * fftmag[223] + melcoe[445] * fftmag[224] +
               melcoe[446] * fftmag[225] + melcoe[447] * fftmag[226] +
               melcoe[448] * fftmag[227] + melcoe[449] * fftmag[228] +
               melcoe[450] * fftmag[229] + melcoe[451] * fftmag[230] +
               melcoe[452] * fftmag[231];
    dout[77] = melcoe[453] * fftmag[224] + melcoe[454] * fftmag[225] +
               melcoe[455] * fftmag[226] + melcoe[456] * fftmag[227] +
               melcoe[457] * fftmag[228] + melcoe[458] * fftmag[229] +
               melcoe[459] * fftmag[230] + melcoe[460] * fftmag[231] +
               melcoe[461] * fftmag[232] + melcoe[462] * fftmag[233] +
               melcoe[463] * fftmag[234] + melcoe[464] * fftmag[235] +
               melcoe[465] * fftmag[236] + melcoe[466] * fftmag[237] +
               melcoe[467] * fftmag[238] + melcoe[468] * fftmag[239];
    dout[78] = melcoe[469] * fftmag[232] + melcoe[470] * fftmag[233] +
               melcoe[471] * fftmag[234] + melcoe[472] * fftmag[235] +
               melcoe[473] * fftmag[236] + melcoe[474] * fftmag[237] +
               melcoe[475] * fftmag[238] + melcoe[476] * fftmag[239] +
               melcoe[477] * fftmag[240] + melcoe[478] * fftmag[241] +
               melcoe[479] * fftmag[242] + melcoe[480] * fftmag[243] +
               melcoe[481] * fftmag[244] + melcoe[482] * fftmag[245] +
               melcoe[483] * fftmag[246] + melcoe[484] * fftmag[247];
    dout[79] = melcoe[485] * fftmag[240] + melcoe[486] * fftmag[241] +
               melcoe[487] * fftmag[242] + melcoe[488] * fftmag[243] +
               melcoe[489] * fftmag[244] + melcoe[490] * fftmag[245] +
               melcoe[491] * fftmag[246] + melcoe[492] * fftmag[247] +
               melcoe[493] * fftmag[248] + melcoe[494] * fftmag[249] +
               melcoe[495] * fftmag[250] + melcoe[496] * fftmag[251] +
               melcoe[497] * fftmag[252] + melcoe[498] * fftmag[253] +
               melcoe[499] * fftmag[254] + melcoe[500] * fftmag[255];
    global_cmvn(dout);
}

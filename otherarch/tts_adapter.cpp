#include "model_adapter.h"
#include "otherarch/utils.h"

#include "common.h"
#include "sampling.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include "src/llama-context.h"

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

struct wav_header {
    char riff[4] = {'R', 'I', 'F', 'F'};
    uint32_t chunk_size;
    char wave[4] = {'W', 'A', 'V', 'E'};
    char fmt[4] = {'f', 'm', 't', ' '};
    uint32_t fmt_chunk_size = 16;
    uint16_t audio_format = 1; // PCM
    uint16_t num_channels = 1; // Mono
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample = 16;
    char data[4] = {'d', 'a', 't', 'a'};
    uint32_t data_size;
};

static std::string save_wav16_base64(const std::vector<float> &data, int sample_rate) {
    std::ostringstream oss;
    wav_header header;

    // Fill header fields
    header.sample_rate = sample_rate;
    header.byte_rate = header.sample_rate * header.num_channels * (header.bits_per_sample / 8);
    header.block_align = header.num_channels * (header.bits_per_sample / 8);
    header.data_size = data.size() * (header.bits_per_sample / 8);
    header.chunk_size = 36 + header.data_size;

    // Write header
    oss.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // Write samples
    for (const auto &sample : data) {
        int16_t pcm_sample = static_cast<int16_t>(std::clamp(sample * 32767.0, -32768.0, 32767.0));
        oss.write(reinterpret_cast<const char*>(&pcm_sample), sizeof(pcm_sample));
    }

    // Get binary WAV data
    std::string wav_data = oss.str();
    return kcpp_base64_encode(wav_data); //return as base64 string
}

static void fill_hann_window(int length, bool periodic, float * output) {
    int offset = -1;
    if (periodic) {
        offset = 0;
    }
    for (int i = 0; i < length; i++) {
        output[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));
    }
}

// very poor-man fft
static void twiddle(float * real, float * imag, int k, int N) {
    float angle = 2 * M_PI * k / N;
    *real = cos(angle);
    *imag = sin(angle);
}

static void irfft(int n, const float * inp_cplx, float * out_real) {
    int N = n / 2 + 1;

    std::vector<float> real_input(N);
    std::vector<float> imag_input(N);
    for (int i = 0; i < N; ++i) {
        real_input[i] = inp_cplx[2 * i];
        imag_input[i] = inp_cplx[2 * i + 1];
    }

    std::vector<float> real_output(n);
    std::vector<float> imag_output(n);

    for (int k = 0; k < n; ++k) {
        real_output[k] = 0.0f;
        imag_output[k] = 0.0f;
        for (int m = 0; m < N; ++m) {
            float twiddle_real;
            float twiddle_imag;

            twiddle(&twiddle_real, &twiddle_imag, k * m, n);

            real_output[k] += real_input[m] * twiddle_real - imag_input[m] * twiddle_imag;
            imag_output[k] += real_input[m] * twiddle_imag + imag_input[m] * twiddle_real;
        }
    }

    for (int i = 0; i < n; ++i) {
        out_real[i] = real_output[i] / N;
    }
}


static void fold(const std::vector<float> & data, int64_t n_out, int64_t n_win, int64_t n_hop, int64_t n_pad, std::vector<float> & output) {
    int64_t output_height = n_out;
    int64_t kernel_w = n_win;
    int64_t stride_w = n_hop;
    int64_t width    = n_out;

    output.resize(width, 0.0f);

    int64_t col_idx = 0;
    for (int64_t w_col = 0; w_col < width; ++w_col) {
        int64_t start = w_col * stride_w - n_pad;
        int64_t end   = start + kernel_w;

        for (int64_t w_im = start; w_im < end; ++w_im) {
            if (w_im >= 0 && w_im < output_height && col_idx < (int64_t) data.size()) {
                output[w_im] += data[col_idx];
            }
            col_idx++;
        }
    }

    output.resize(n_out - 2 * n_pad);
}

// TODO: not optimized at all
static std::vector<float> embd_to_audio(
        const float * embd,
        const int n_codes,
        const int n_embd,
        const int n_thread) {
    const int n_fft = 1280;
    const int n_hop = 320;
    const int n_win = 1280;
    const int n_pad = (n_win - n_hop)/2;
    const int n_out = (n_codes - 1)*n_hop + n_win;

    std::vector<float> hann(n_fft);

    fill_hann_window(hann.size(), true, hann.data());

    int n_spec = n_embd*n_codes;

    std::vector<float> E (n_spec);
    std::vector<float> S (n_spec);
    std::vector<float> ST(n_spec);

    for (int l = 0; l < n_codes; ++l) {
        for (int k = 0; k < n_embd; ++k) {
            E[k*n_codes + l] = embd[l*n_embd + k];
        }
    }

    for (int k = 0; k < n_embd/2; ++k) {
        for (int l = 0; l < n_codes; ++l) {
            float mag = E[(k           )*n_codes + l];
            float phi = E[(k + n_embd/2)*n_codes + l];

            mag = exp(mag);

            if (mag > 1e2) {
                mag = 1e2;
            }
            S[2*(k*n_codes + l) + 0] = mag*cosf(phi);
            S[2*(k*n_codes + l) + 1] = mag*sinf(phi);
        }
    }

    for (int l = 0; l < n_codes; ++l) {
        for (int k = 0; k < n_embd/2; ++k) {
            ST[l*n_embd + 2*k + 0] = S[2*(k*n_codes + l) + 0];
            ST[l*n_embd + 2*k + 1] = S[2*(k*n_codes + l) + 1];
        }
    }

    std::vector<float> res  (n_codes*n_fft);
    std::vector<float> hann2(n_codes*n_fft);

    std::vector<std::thread> workers(n_thread);
    for (int i = 0; i < n_thread; ++i) {
        workers[i] = std::thread([&, i]() {
            for (int l = i; l < n_codes; l += n_thread) {
                irfft(n_fft, ST.data() + l*n_embd, res.data() + l*n_fft);
                for (int j = 0; j < n_fft; ++j) {
                    res  [l*n_fft + j] *= hann[j];
                    hann2[l*n_fft + j]  = hann[j] * hann[j];
                }
            }
        });
    }
    for (int i = 0; i < n_thread; ++i) {
        workers[i].join();
    }

    std::vector<float> audio;
    std::vector<float> env;

    fold(res,   n_out, n_win, n_hop, n_pad, audio);
    fold(hann2, n_out, n_win, n_hop, n_pad, env); // TODO: can be done once

    for (size_t i = 0; i < audio.size(); ++i) {
        audio[i] /= env[i];
    }

    return audio;
}

static const std::map<int, std::string> ones = {
    {0, "zero"}, {1, "one"}, {2, "two"}, {3, "three"}, {4, "four"},
    {5, "five"}, {6, "six"}, {7, "seven"}, {8, "eight"}, {9, "nine"},
    {10, "ten"}, {11, "eleven"}, {12, "twelve"}, {13, "thirteen"}, {14, "fourteen"},
    {15, "fifteen"}, {16, "sixteen"}, {17, "seventeen"}, {18, "eighteen"}, {19, "nineteen"}
};

static const std::map<int, std::string> tens = {
    {2, "twenty"}, {3, "thirty"}, {4, "forty"}, {5, "fifty"},
    {6, "sixty"}, {7, "seventy"}, {8, "eighty"}, {9, "ninety"}
};

// Convert a number less than 1000 to words
static std::string convert_less_than_thousand(int num) {
    std::string result;

    if (num >= 100) {
        result += ones.at(num / 100) + " hundred ";
        num %= 100;
    }

    if (num >= 20) {
        result += tens.at(num / 10);
        if (num % 10 > 0) {
            result += "-" + ones.at(num % 10);
        }
    } else if (num > 0) {
        result += ones.at(num);
    }

    return result;
}

static std::string number_to_words(const std::string & number_str) {
    try {
        size_t decimal_pos = number_str.find('.');
        std::string integer_part = number_str.substr(0, decimal_pos);

        int int_number = std::stoi(integer_part);
        std::string result;

        if (int_number == 0) {
            result = "zero";
        } else {
            if (int_number >= 1000000000) {
                int billions = int_number / 1000000000;
                result += convert_less_than_thousand(billions) + " billion ";
                int_number %= 1000000000;
            }

            if (int_number >= 1000000) {
                int millions = int_number / 1000000;
                result += convert_less_than_thousand(millions) + " million ";
                int_number %= 1000000;
            }

            if (int_number >= 1000) {
                int thousands = int_number / 1000;
                result += convert_less_than_thousand(thousands) + " thousand ";
                int_number %= 1000;
            }

            if (int_number > 0) {
                result += convert_less_than_thousand(int_number);
            }
        }

        // Handle decimal part
        if (decimal_pos != std::string::npos) {
            result += " point";
            std::string decimal_part = number_str.substr(decimal_pos + 1);
            for (char digit : decimal_part) {
                result += " " + ones.at(digit - '0');
            }
        }

        return result;
    } catch (const std::exception& e) {
        // Skip if fails
        return " ";
    }
}

static std::string replace_numbers_with_words(const std::string & input_text) {
    std::regex number_pattern(R"(\d+(\.\d+)?)");
    std::string result;
    auto it = std::sregex_iterator(input_text.begin(), input_text.end(), number_pattern);
    auto end = std::sregex_iterator();

    size_t last_pos = 0;
    for (std::sregex_iterator i = it; i != end; ++i) {
        const std::smatch& match = *i;
        result.append(input_text, last_pos, match.position() - last_pos);
        result.append(number_to_words(match.str()));
        last_pos = match.position() + match.length();
    }
    result.append(input_text, last_pos);

    return result;
}

static std::string process_text(const std::string & text) {

    std::string processed_text = replace_numbers_with_words(text);

    std::transform(processed_text.begin(), processed_text.end(),
                  processed_text.begin(), ::tolower);

    std::regex special_chars(R"([-_/,\.\\])");
    processed_text = std::regex_replace(processed_text, special_chars, " ");
    std::regex non_alpha(R"([^a-z\s])");
    processed_text = std::regex_replace(processed_text, non_alpha, "");
    std::regex multiple_spaces(R"(\s+)");
    processed_text = std::regex_replace(processed_text, multiple_spaces, " ");
    processed_text = std::regex_replace(processed_text, std::regex(R"(^\s+|\s+$)"), "");
    processed_text = std::regex_replace(processed_text, std::regex(R"(\s)"), "<|text_sep|>");

    return processed_text;
}


static void prompt_add(llama_tokens & prompt, const llama_tokens & tokens) {
    prompt.insert(prompt.end(), tokens.begin(), tokens.end());
}
static void prompt_add(llama_tokens & prompt, const llama_model * model, const std::string & txt, bool add_special, bool parse_special) {
    auto tmp = common_tokenize(model, txt, add_special, parse_special);
    prompt_add(prompt, tmp);
}
static void prompt_init(llama_tokens & prompt, const llama_model * model) {
    prompt.clear();
    prompt_add(prompt, model, "<|im_start|>\n", true, true);
}

static std::vector<llama_token> prepare_guide_tokens(const llama_model * model, const std::string& str)
{
    const std::string& delimiter = "<|text_sep|>";

    std::vector<llama_token> result;
    size_t start = 0;
    size_t end = str.find(delimiter);

    while (end != std::string::npos) {
        std::string current_word = str.substr(start, end - start);
        auto tmp = common_tokenize(model, current_word, false, true);
        result.push_back(tmp[0]);
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }

    // Add the last part
    std::string current_word = str.substr(start);
    auto tmp = common_tokenize(model, current_word, false, true);
    result.push_back(tmp[0]);
    return result;
}

static llama_context * ttc_ctx = nullptr; //text to codes ctx
static llama_context * cts_ctx = nullptr; //codes to speech

static int ttsdebugmode = 0;
static std::string ttsplatformenv, ttsdeviceenv, ttsvulkandeviceenv;
bool ttstype_load_model(const tts_load_model_inputs inputs)
{
    //duplicated from expose.cpp
    int cl_parseinfo = inputs.clblast_info; //first digit is whether configured, second is platform, third is devices
    std::string usingclblast = "GGML_OPENCL_CONFIGURED="+std::to_string(cl_parseinfo>0?1:0);
    putenv((char*)usingclblast.c_str());
    cl_parseinfo = cl_parseinfo%100; //keep last 2 digits
    int platform = cl_parseinfo/10;
    int devices = cl_parseinfo%10;
    ttsplatformenv = "GGML_OPENCL_PLATFORM="+std::to_string(platform);
    ttsdeviceenv = "GGML_OPENCL_DEVICE="+std::to_string(devices);
    putenv((char*)ttsplatformenv.c_str());
    putenv((char*)ttsdeviceenv.c_str());
    std::string vulkan_info_raw = inputs.vulkan_info;
    std::string vulkan_info_str = "";
    for (size_t i = 0; i < vulkan_info_raw.length(); ++i) {
        vulkan_info_str += vulkan_info_raw[i];
        if (i < vulkan_info_raw.length() - 1) {
            vulkan_info_str += ",";
        }
    }
    if(vulkan_info_str!="")
    {
        ttsvulkandeviceenv = "GGML_VK_VISIBLE_DEVICES="+vulkan_info_str;
        putenv((char*)ttsvulkandeviceenv.c_str());
    }


    std::string modelfile_ttc = inputs.ttc_model_filename;
    std::string modelfile_cts = inputs.cts_model_filename;
    printf("\nLoading TTS Model, OuteTTS: %s, WavTokenizer: %s",modelfile_ttc.c_str(),modelfile_cts.c_str());

    ttsdebugmode = inputs.debugmode;

    // tts init
    llama_model_params tts_model_params = llama_model_default_params();
    llama_context_params tts_ctx_params = llama_context_default_params();

    tts_model_params.use_mmap = false;
    tts_model_params.use_mlock = false;
    tts_model_params.n_gpu_layers = 999; //offload if possible
    tts_model_params.split_mode = llama_split_mode::LLAMA_SPLIT_MODE_LAYER;
    tts_ctx_params.n_ctx = 8192;
    tts_ctx_params.logits_all = false;
    tts_ctx_params.offload_kqv = true;
    tts_ctx_params.n_batch = 8192;
    tts_ctx_params.n_ubatch = 512;
    tts_ctx_params.n_threads = 4;
    tts_ctx_params.n_threads_batch = 4;
    tts_ctx_params.flash_attn = false;
    tts_ctx_params.embeddings = true;

    llama_backend_init();

    llama_model * ttcmodel = llama_model_load_from_file(modelfile_ttc.c_str(), tts_model_params);
    ttc_ctx = llama_new_context_with_model(ttcmodel, tts_ctx_params);

    if (ttc_ctx == nullptr) {
        printf("\nTTS Load Error: Failed to initialize ttc context!\n");
        return false;
    }

    llama_model * ctsmodel = llama_model_load_from_file(modelfile_cts.c_str(), tts_model_params);
    cts_ctx = llama_new_context_with_model(ctsmodel, tts_ctx_params);

    if (cts_ctx == nullptr) {
        printf("\nTTS Load Error: Failed to initialize cts context!\n");
        return false;
    }

    printf("\nTTS Load Complete.\n");
    return true;
}

tts_generation_outputs ttstype_generate(const tts_generation_inputs inputs)
{
    // tts_generation_outputs output;

    // if(ttc_ctx==nullptr || cts_ctx==nullptr)
    // {
    //     printf("\nWarning: KCPP TTS not initialized!\n");
    //     output.data = nullptr;
    //     output.status = 0;
    //     return output;
    // }

    // if(!inputs.quiet)
    // {
    //     printf("\nTTS Generating...");
    // }

    // std::vector<llama_token> codes;
    // std::vector<llama_token> guide_tokens;
    // const llama_model * model_ttc = &(ttc_ctx->model);
    // std::string prompt = inputs.prompt;

    // // process prompt and generate voice codes
    // {
    //     std::vector<llama_token> prompt_inp;
    //     prompt_init(prompt_inp, model_ttc);
    //     prompt_add(prompt_inp, model_ttc, "<|text_start|>", false, true);

    //     //add the speaker based on the seed
    //     if(inputs.speaker_seed>0)
    //     {
    //         std::string sampletext = "but<|text_sep|>that<|text_sep|>is<|text_sep|>what<|text_sep|>it<|text_sep|>is<|text_sep|>";
    //     }

    //     // convert the input text into the necessary format expected by OuteTTS
    //     std::string prompt_clean = process_text(prompt);
    //     guide_tokens = prepare_guide_tokens(model_ttc,prompt_clean);
    //     prompt_add(prompt_inp, model_ttc, prompt_clean, false, true);

    //     prompt_add(prompt_inp, model_ttc, "<|text_end|>\n", false, true);

    //     //create batch with tokens for decoding prompt processing
    //     llama_kv_cache_clear(ttc_ctx);
    //     llama_kv_cache_clear(cts_ctx);
    //     llama_batch batch = llama_batch_get_one(prompt_inp.data(), prompt_inp.size());

    //     if (llama_decode(ttc_ctx, batch) != 0) {
    //         printf("\nError: TTS prompt batch processing failed\n");
    //         output.data = nullptr;
    //         output.status = 0;
    //         return output;
    //     }

    //     // main loop
    //     int n_past   = batch.n_tokens;
    //     int n_decode = 0;
    //     int n_predict = 4096; //max 4096 tokens

    //     bool next_token_uses_guide_token = true;

    //     while (n_decode <= n_predict)
    //     {
    //         float * logits = llama_get_logits(ttc_ctx);

    //         llama_token new_token_id = common_sampler_sample(smpl[i], ctx_ttc, i_batch[i]);

    //             //guide tokens help prevent hallucinations by forcing the TTS to use the correct word
    //             if(!guide_tokens.empty() && next_token_uses_guide_token && !llama_token_is_control(model_ttc, new_token_id) && !llama_token_is_eog(model_ttc, new_token_id))
    //             {
    //                 llama_token guide_token = guide_tokens[0];
    //                 guide_tokens.erase(guide_tokens.begin());
    //                 new_token_id = guide_token; //ensure correct word fragment is used
    //             }

    //             //this is the token id that always precedes a new word
    //             next_token_uses_guide_token = (new_token_id == 198);

    //             common_sampler_accept(smpl[i], new_token_id, true);

    //             codes.push_back(new_token_id);

    //             const auto * cands = common_sampler_get_candidates(smpl[i]);

    //             // is it an end of generation? -> mark the stream as finished
    //             if (llama_token_is_eog(model_ttc, new_token_id) || n_decode == n_predict) {
    //                 std::string reason;
    //                 if (llama_token_is_eog(model_ttc, new_token_id)) {
    //                     reason = "eos";
    //                 } else {
    //                     reason = "n_predict";
    //                 }

    //                 i_batch[i] = -1;

    //                 LOG("\n");
    //                 if (n_parallel > 1) {
    //                     LOG_CNT("\n");
    //                     LOG_INF("%s: stream %d finished at n_past = %d, reason = '%s'\n", __func__, i, n_past, reason.c_str());
    //                 }

    //                 continue;
    //             }

    //             {
    //                 const float p = cands->data[cands->selected].p;

    //                 const int col = std::max(0, std::min((int) k_colors.size() - 1, (int) ((3*p)*float(k_colors.size()))));

    //                 LOG_CNT("%s%d%s", k_colors[col].c_str(), i, "\033[0m");
    //                 //LOG_CNT("%d", i);
    //             }

    //             i_batch[i] = batch.n_tokens;

    //             // push this new token for next evaluation
    //             common_batch_add(batch, new_token_id, n_past, { i }, true);
    //         }

    //         // all streams are finished
    //         if (batch.n_tokens == 0) {
    //             break;
    //         }

    //         n_decode += 1;
    //         n_past += 1;

    //         // evaluate the current batch with the transformer model
    //         if (llama_decode(ctx_ttc, batch)) {
    //             LOG_ERR("%s : failed to eval, return code %d\n", __func__, 1);
    //             return 1;
    //         }
    //     }

    //     llama_batch_free(batch);

    //     LOG("\n");
    //     LOG_INF("%s: time for decoder:       %.3f ms\n", __func__, (ggml_time_us() - t_dec_start) / 1000.0f);




    // const std::string inp_txt = common_detokenize(ctx_ttc, codes, true);
    // printf("codes (size %d): '%s'\n",(int) codes.size(), inp_txt.c_str());

    // // remove all non-audio tokens (i.e. < 151672 || > 155772)
    // codes.erase(std::remove_if(codes.begin(), codes.end(), [](llama_token t) { return t < 151672 || t > 155772; }), codes.end());

    // for (auto & token : codes) {
    //     token -= 151672;
    // }

    // const auto t_voc_start = ggml_time_us();
    // const int n_codes = codes.size();

    // llama_batch batch = llama_batch_init(n_codes, 0, 1);

    // for (size_t i = 0; i < codes.size(); ++i) {
    //     common_batch_add(batch, codes[i], i, { 0 }, true); // TODO: all logits?
    // }
    // GGML_ASSERT(batch.n_tokens == n_codes);

    // if (llama_decode(ctx_cts, batch) != 0) {
    //     LOG_ERR("%s: llama_decode() failed\n", __func__);
    //     return 1;
    // }

    // llama_synchronize(ctx_cts);

    // LOG_INF("%s: time for vocoder:      %.3f ms\n", __func__, (ggml_time_us() - t_voc_start) / 1000.0f);

    // const auto t_spec_start = ggml_time_us();

    // // spectral operations
    // const int n_embd = llama_n_embd(model_cts);
    // const float * embd = llama_get_embeddings(ctx_cts);

    // auto audio = embd_to_audio(embd, n_codes, n_embd, params.cpuparams.n_threads);



    // const std::string fname = "output.wav";

    // const int n_sr = 24000; // sampling rate

    // // zero out first 0.25 seconds
    // for (int i = 0; i < 24000/4; ++i) {
    //     audio[i] = 0.0f;
    // }

    // LOG_INF("%s: time for spectral ops: %.3f ms\n", __func__, (ggml_time_us() - t_spec_start) / 1000.0f);
    // LOG_INF("%s: total time:            %.3f ms\n", __func__, (ggml_time_us() - t_main_start) / 1000.0f);

    // save_wav16(fname, audio, n_sr);

    // LOG_INF("%s: audio written to file '%s'\n", __func__, fname.c_str());

    // llama_backend_free();

    // return 0;






}

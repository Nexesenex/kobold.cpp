#include <stdio.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "model_adapter.h"

#include "stable-diffusion.cpp"
#include "util.cpp"
#include "upscaler.cpp"
#include "model.cpp"
#include "zip.c"

// #include "preprocessing.hpp"
#include "stable-diffusion.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

const char* rng_type_to_str[] = {
    "std_default",
    "cuda",
};

// Names of the sampler method, same order as enum sample_method in stable-diffusion.h
const char* sample_method_str[] = {
    "euler_a",
    "euler",
    "heun",
    "dpm2",
    "dpm++2s_a",
    "dpm++2m",
    "dpm++2mv2",
    "lcm",
};

// Names of the sigma schedule overrides, same order as sample_schedule in stable-diffusion.h
const char* schedule_str[] = {
    "default",
    "discrete",
    "karras",
};

const char* modes_str[] = {
    "txt2img",
    "img2img",
    "img2vid",
    "convert",
};

enum SDMode {
    TXT2IMG,
    IMG2IMG,
    IMG2VID,
    CONVERT,
    MODE_COUNT
};

struct SDParams {
    int n_threads = -1;
    SDMode mode   = TXT2IMG;

    std::string model_path;
    std::string vae_path;
    std::string taesd_path;
    std::string esrgan_path;
    std::string controlnet_path;
    std::string embeddings_path;
    sd_type_t wtype = SD_TYPE_COUNT;
    std::string lora_model_dir;
    std::string output_path = "output.png";
    std::string input_path;
    std::string control_image_path;

    std::string prompt;
    std::string negative_prompt;
    float min_cfg   = 1.0f;
    float cfg_scale = 7.0f;
    int clip_skip   = -1;  // <= 0 represents unspecified
    int width       = 512;
    int height      = 512;
    int batch_count = 1;

    int video_frames         = 6;
    int motion_bucket_id     = 127;
    int fps                  = 6;
    float augmentation_level = 0.f;

    sample_method_t sample_method = EULER_A;
    schedule_t schedule           = DEFAULT;
    int sample_steps              = 20;
    float strength                = 0.75f;
    float control_strength        = 0.9f;
    rng_type_t rng_type           = CUDA_RNG;
    int64_t seed                  = 42;
    bool verbose                  = false;
    bool vae_tiling               = false;
    bool control_net_cpu          = false;
    bool canny_preprocess         = false;
    int upscale_repeats           = 1;
};

//global static vars for SD
static SDParams * sd_params = nullptr;
static sd_ctx_t * sd_ctx = nullptr;

static void sd_logger_callback(enum sd_log_level_t level, const char* log, void* data) {
    SDParams* params = (SDParams*)data;
    if (!params->verbose && level <= SD_LOG_DEBUG) {
        return;
    }
    if (level <= SD_LOG_INFO) {
        fputs(log, stdout);
        fflush(stdout);
    } else {
        fputs(log, stderr);
        fflush(stderr);
    }
}

bool sdtype_load_model(const sd_load_model_inputs inputs) {

    printf("\nImage Gen - Load Safetensors Image Model: %s\n",inputs.model_filename);

    sd_params = new SDParams();
    sd_params->model_path = inputs.model_filename;
    sd_params->wtype = SD_TYPE_F16;
    sd_params->n_threads = -1; //use physical cores
    sd_params->input_path = ""; //unused

    if(inputs.debugmode==1)
    {
        sd_set_log_callback(sd_logger_callback, (void*)sd_params);
    }

    bool vae_decode_only = false;

    sd_ctx = new_sd_ctx(sd_params->model_path.c_str(),
                                  sd_params->vae_path.c_str(),
                                  sd_params->taesd_path.c_str(),
                                  sd_params->controlnet_path.c_str(),
                                  sd_params->lora_model_dir.c_str(),
                                  sd_params->embeddings_path.c_str(),
                                  vae_decode_only,
                                  sd_params->vae_tiling,
                                  true,
                                  sd_params->n_threads,
                                  sd_params->wtype,
                                  sd_params->rng_type,
                                  sd_params->schedule,
                                  sd_params->control_net_cpu);

    if (sd_ctx == NULL) {
        printf("\nError: KCPP SD Failed to create context!\n");
        return false;
    }

    return true;

}

sd_generation_outputs sdtype_generate(const sd_generation_inputs inputs)
{
    sd_generation_outputs output;
    if(sd_ctx == nullptr || sd_params == nullptr)
    {
        printf("\nError: KCPP SD is not initialized!\n");
        output.data = nullptr;
        output.status = 0;
        output.data_length = 0;
        return output;
    }
    uint8_t * input_image_buffer = NULL;
    sd_image_t * results;
    sd_image_t* control_image = NULL;

    sd_params->prompt = inputs.prompt;
    sd_params->negative_prompt = inputs.negative_prompt;
    sd_params->cfg_scale = inputs.cfg_scale;
    sd_params->sample_steps = inputs.sample_steps;
    sd_params->seed = inputs.seed;

    printf("\nGenerating Image (%d steps)\n",inputs.sample_steps);
    std::string sampler = inputs.sample_method;

    if(sampler=="euler a") //all lowercase
    {
        sd_params->sample_method = sample_method_t::EULER_A;
    }
    else if(sampler=="euler")
    {
        sd_params->sample_method = sample_method_t::EULER;
    }
    else if(sampler=="heun")
    {
        sd_params->sample_method = sample_method_t::HEUN;
    }
    else if(sampler=="dpm2")
    {
        sd_params->sample_method = sample_method_t::DPM2;
    }
    else if(sampler=="dpm++ 2m karras" || sampler=="dpm++ 2m")
    {
        sd_params->sample_method = sample_method_t::DPMPP2M;
    }
    else
    {
        sd_params->sample_method = sample_method_t::EULER_A;
    }

    if (sd_params->mode == TXT2IMG) {
        results = txt2img(sd_ctx,
                          sd_params->prompt.c_str(),
                          sd_params->negative_prompt.c_str(),
                          sd_params->clip_skip,
                          sd_params->cfg_scale,
                          sd_params->width,
                          sd_params->height,
                          sd_params->sample_method,
                          sd_params->sample_steps,
                          sd_params->seed,
                          sd_params->batch_count,
                          control_image,
                          sd_params->control_strength);
    } else {
        sd_image_t input_image = {(uint32_t)sd_params->width,
                                  (uint32_t)sd_params->height,
                                  3,
                                  input_image_buffer};
        results = img2img(sd_ctx,
                            input_image,
                            sd_params->prompt.c_str(),
                            sd_params->negative_prompt.c_str(),
                            sd_params->clip_skip,
                            sd_params->cfg_scale,
                            sd_params->width,
                            sd_params->height,
                            sd_params->sample_method,
                            sd_params->sample_steps,
                            sd_params->strength,
                            sd_params->seed,
                            sd_params->batch_count);
    }

    if (results == NULL) {
        printf("\nKCPP SD generate failed!\n");
        output.data = nullptr;
        output.status = 0;
        output.data_length = 0;
        return output;
    }


    size_t last            = sd_params->output_path.find_last_of(".");
    std::string dummy_name = last != std::string::npos ? sd_params->output_path.substr(0, last) : sd_params->output_path;
    for (int i = 0; i < sd_params->batch_count; i++) {
        if (results[i].data == NULL) {
            continue;
        }
        std::string final_image_path = i > 0 ? dummy_name + "_" + std::to_string(i + 1) + ".png" : dummy_name + ".png";
        stbi_write_png(final_image_path.c_str(), results[i].width, results[i].height, results[i].channel,
                       results[i].data, 0, "Made By KoboldCpp");
        printf("save result image to '%s'\n", final_image_path.c_str());
        free(results[i].data);
        results[i].data = NULL;
    }

    free(results);

    output.data = nullptr;
    output.status = 1;
    output.data_length = 0;
    return output;
}

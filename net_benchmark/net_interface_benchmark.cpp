#include <float.h>
#include <stdio.h>
#include <string>

#include "benchmark.h"
#include "cpu.h"
#include "datareader.h"
#include "net.h"
#include "gpu.h"

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;
static int g_warmup_loop_count = 8;
static int g_loop_count = 10;

double time_min = DBL_MAX;
double time_max = -DBL_MAX;
double time_avg = 0;
std::string ncnn_param_file = "yolo-fastestv2-anchorfree.ncnn.param";
std::string ncnn_bin_file = "yolo-fastestv2-anchorfree.ncnn.bin";
std::string input_name = "input.1";    // in0  input.1"
std::string output_name1 = "789";      // out0  789
std::string output_name2 = "790";      // out1  790
int main(int argc, char **argv)
{
    int loop_count = 50;
    int num_threads = 1;
    int powersave = 0;
    int gpu_device = -1;

    if (argc >= 2) {
        num_threads = atoi(argv[1]);
    }
    if (argc >= 3) {
        loop_count = atoi(argv[2]);
    }
    if (argc >= 4) {
        ncnn_param_file = argv[3];
    }
    if (argc >= 5) {
        ncnn_bin_file = argv[4];
    }
    if (argc >= 6) {
        input_name = argv[5];
    }
    if (argc >= 7) {
        output_name1 = argv[6];
    }
    if (argc >= 8) {
        output_name2 = argv[7];
    }
    
    g_blob_pool_allocator.set_size_compare_ratio(0.0f);
    g_workspace_pool_allocator.set_size_compare_ratio(0.5f);

    g_loop_count = loop_count;
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = num_threads;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;

    opt.use_winograd_convolution = true;
    opt.use_sgemm_convolution = true;
    opt.use_int8_inference = false;
    opt.use_vulkan_compute = false;
    opt.use_bf16_storage = true;
    // opt.use_fp16_packed = true;   
    // opt.use_fp16_storage = true;
    // opt.use_fp16_arithmetic = true;
    opt.use_int8_storage = true;
    opt.use_int8_arithmetic = true;
    opt.use_packing_layout = true;
    opt.use_shader_pack8 = false;
    opt.use_image_storage = false;

    ncnn::set_cpu_powersave(0);
    ncnn::set_omp_dynamic(0);
    // ncnn::set_omp_num_threads(num_threads);

    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();
    
    fprintf(stderr, "num_threads = %d\n", num_threads);
    fprintf(stderr, "loop_count = %d\n", g_loop_count);
    fprintf(stderr, "powersave = %d\n", ncnn::get_cpu_powersave());
    fprintf(stderr, "gpu_device = %d\n", gpu_device);
    fprintf(stderr, "param file = %s\n", ncnn_param_file.c_str());
    fprintf(stdout, "bin file = %s\n", ncnn_bin_file.c_str());
    ncnn::Mat in = ncnn::Mat(352, 352, 3);


    in.fill(0.01f);

    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();

    ncnn::Net net;

    net.opt = opt;

    net.load_param(ncnn_param_file.c_str());
    net.load_model(ncnn_bin_file.c_str());

    
    ncnn::Mat out1;
    ncnn::Mat out2;
    
    // warm up
    for (int i = 0; i < g_warmup_loop_count; i++)
    {
        ncnn::Extractor ex = net.create_extractor();
        ex.input(input_name.c_str(), in);
        
        ex.extract(output_name1.c_str(), out1);
        ex.extract(output_name2.c_str(), out2);    
    }

    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;

    for (int i = 0; i < g_loop_count; i++)
    {
        double start = ncnn::get_current_time();

        {
            ncnn::Extractor ex = net.create_extractor();
            ex.input(input_name.c_str(), in);
        
            ex.extract(output_name1.c_str(), out1);
            ex.extract(output_name2.c_str(), out2);  
        }

        double end = ncnn::get_current_time();

        double time = end - start;

        time_min = std::min(time_min, time);
        time_max = std::max(time_max, time);
        time_avg += time;
    }

    time_avg /= g_loop_count;

    fprintf(stderr, "  min = %7.2f  max = %7.2f  avg = %7.2f\n", time_min, time_max, time_avg);

}

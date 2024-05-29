#pragma once
#if NVJPEG_FOUND
#include <torch/types.h>
#include <nvjpeg.h>

namespace vision {
namespace image {

class CUDAJpegEncoder
{
 public:
  CUDAJpegEncoder(const torch::Device& device);
  ~CUDAJpegEncoder();

  torch::Tensor encode_jpeg(
    const torch::Tensor& src_image,
    const torch::Device& device);

  void setQuality(const int64_t);

  torch::Device device;

 protected:
  cudaStream_t stream;
  nvjpegEncoderState_t nv_enc_state;
  nvjpegEncoderParams_t nv_enc_params;
  nvjpegHandle_t nvjpeg_handle;

};
}}
#endif

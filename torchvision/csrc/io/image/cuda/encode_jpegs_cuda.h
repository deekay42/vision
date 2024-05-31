#pragma once
#include <vector>
#include <torch/types.h>
#if NVJPEG_FOUND
#include <c10/cuda/CUDAStream.h>
#include <nvjpeg.h>

namespace vision {
namespace image {

class CUDAJpegEncoder {
 public:
  CUDAJpegEncoder(const torch::Device& device);
  ~CUDAJpegEncoder();

  torch::Tensor encode_jpeg(
      const torch::Tensor& src_image,
      const torch::Device& device);

  void setQuality(const int64_t);

  const torch::Device original_device;
  const torch::Device target_device;
  const c10::cuda::CUDAStream stream;

 protected:
  // dedicated stream to avoid interfering with calling code stream(s)
  nvjpegEncoderState_t nv_enc_state;
  nvjpegEncoderParams_t nv_enc_params;
  nvjpegHandle_t nvjpeg_handle;
};
} // namespace image
} // namespace vision
#endif

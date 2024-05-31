#pragma once
#include <vector>
#include <torch/types.h>
#include "../image_read_mode.h"

#if NVJPEG_FOUND
#include <nvjpeg.h>

namespace vision {
namespace image {
class CUDAJpegDecoder {
 public:
  CUDAJpegDecoder();
  ~CUDAJpegDecoder();

  std::vector<torch::Tensor> decode_images(
      const std::vector<torch::Tensor>& encoded_images,
      const nvjpegOutputFormat_t& output_format,
      const torch::Device& device);

 protected:
  std::tuple<
      std::vector<nvjpegImage_t>,
      std::vector<torch::Tensor>,
      std::vector<int>>
  prepare_buffers(
      const std::vector<torch::Tensor>& encoded_images,
      const nvjpegOutputFormat_t& output_format,
      const torch::Device& device);
  nvjpegJpegState_t nvjpeg_state;
  cudaStream_t stream;
  nvjpegJpegState_t nvjpeg_decoupled_state;
  nvjpegBufferPinned_t pinned_buffers[2];
  nvjpegBufferDevice_t device_buffer;
  nvjpegJpegStream_t jpeg_streams[2];
  nvjpegDecodeParams_t nvjpeg_decode_params;
  nvjpegJpegDecoder_t nvjpeg_decoder;
  bool hw_decode_available{true};
  nvjpegHandle_t nvjpeg_handle;
};
} // namespace image
} // namespace vision
#endif

#include "decode_jpegs_cuda.h"
#if !NVJPEG_FOUND

std::vector<torch::Tensor> decode_jpegs_cuda(
    const std::vector<torch::Tensor>& encoded_images,
    vision::image::ImageReadMode mode,
    torch::Device device) {
  TORCH_CHECK(
      false, "decode_jpegs_cuda: torchvision not compiled with nvJPEG support");
}

#else
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

namespace vision {
namespace image {

std::mutex decoderMutex;
std::unique_ptr<CUDAJpegDecoder> cudaJpegDecoder;
torch::Device activeDecoderDevice(torch::kCPU);

std::vector<torch::Tensor> decode_jpegs_cuda(
    const std::vector<torch::Tensor>& encoded_images,
    vision::image::ImageReadMode mode,
    torch::Device device) {
  C10_LOG_API_USAGE_ONCE(
      "torchvision.csrc.io.image.cuda.nvjpeg.decode_jpegs_cuda");

  // some nvjpeg structures are not thread safe so we're keeping it single
  // threaded for now. in the future this may be an opportunity to unlock
  // further speedups
  std::lock_guard<std::mutex> lock(decoderMutex);
  at::cuda::CUDAGuard device_guard(device);

  // lazy init of the decoder class
  // the decoder holds on to a lot of state and is expensive to create, so we
  // reuse it across calls
  if (cudaJpegDecoder == nullptr || device != activeDecoderDevice) {
    cudaJpegDecoder = std::make_unique<CUDAJpegDecoder>();
    activeDecoderDevice = device;
  }

  for (auto& encoded_image : encoded_images) {
    TORCH_CHECK(
        encoded_image.dtype() == torch::kU8, "Expected a torch.uint8 tensor");

    TORCH_CHECK(
        !encoded_image.is_cuda(),
        "The input tensor must be on CPU when decoding with nvjpeg")

    TORCH_CHECK(
        encoded_image.dim() == 1 && encoded_image.numel() > 0,
        "Expected a non empty 1-dimensional tensor");
  }

  TORCH_CHECK(device.is_cuda(), "Expected a cuda device");

  int major_version;
  int minor_version;
  nvjpegStatus_t get_major_property_status =
      nvjpegGetProperty(MAJOR_VERSION, &major_version);
  nvjpegStatus_t get_minor_property_status =
      nvjpegGetProperty(MINOR_VERSION, &minor_version);

  TORCH_CHECK(
      get_major_property_status == NVJPEG_STATUS_SUCCESS,
      "nvjpegGetProperty failed: ",
      get_major_property_status);
  TORCH_CHECK(
      get_minor_property_status == NVJPEG_STATUS_SUCCESS,
      "nvjpegGetProperty failed: ",
      get_minor_property_status);
  if ((major_version < 11) || ((major_version == 11) && (minor_version < 6))) {
    TORCH_WARN_ONCE(
        "There is a memory leak issue in the nvjpeg library for CUDA versions < 11.6. "
        "Make sure to rely on CUDA 11.6 or above before using decode_jpeg(..., device='cuda').");
  }

  nvjpegOutputFormat_t output_format;

  switch (mode) {
    case vision::image::IMAGE_READ_MODE_UNCHANGED:
      // Using NVJPEG_OUTPUT_UNCHANGED causes differently sized output channels
      // which is related to the subsampling used I'm not sure why this is the
      // case, but for now we're just using RGB and later removing channels from
      // grayscale images.
      output_format = NVJPEG_OUTPUT_UNCHANGED;
      break;
    case vision::image::IMAGE_READ_MODE_GRAY:
      output_format = NVJPEG_OUTPUT_Y;
      break;
    case vision::image::IMAGE_READ_MODE_RGB:
      output_format = NVJPEG_OUTPUT_RGB;
      break;
    default:
      TORCH_CHECK(
          false, "The provided mode is not supported for JPEG decoding on GPU");
  }

  try {
    return cudaJpegDecoder->decode_images(
        encoded_images, output_format, device);
  } catch (const std::exception& e) {
    if (typeid(e) != typeid(std::runtime_error)) {
      TORCH_CHECK(false, "Error while decoding JPEG images: ", e.what());
    } else {
      throw;
    }
  }
}

CUDAJpegDecoder::CUDAJpegDecoder() {
  /*
    Many of nvjpeg's status variables can be reused across calls,
    so we initialize them here and save them as class members
  */

  nvjpegStatus_t status;
  cudaError_t cudaStatus;

  status = nvjpegCreateEx(
      NVJPEG_BACKEND_HARDWARE,
      NULL,
      NULL,
      NVJPEG_FLAGS_DEFAULT,
      &nvjpeg_handle);
  if (status == NVJPEG_STATUS_ARCH_MISMATCH) {
    status = nvjpegCreateEx(
        NVJPEG_BACKEND_DEFAULT,
        NULL,
        NULL,
        NVJPEG_FLAGS_DEFAULT,
        &nvjpeg_handle);
    TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to initialize nvjpeg with default backend: ",
        status);
    hw_decode_available = false;
  } else {
    TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to initialize nvjpeg with hardware backend: ",
        status);
  }

  status = nvjpegJpegStateCreate(nvjpeg_handle, &nvjpeg_state);
  TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create nvjpeg state: ",
      status);

  status = nvjpegDecoderCreate(
      nvjpeg_handle, NVJPEG_BACKEND_DEFAULT, &nvjpeg_decoder);
  TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create nvjpeg decoder: ",
      status);

  status = nvjpegDecoderStateCreate(
      nvjpeg_handle, nvjpeg_decoder, &nvjpeg_decoupled_state);
  TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create nvjpeg decoder state: ",
      status);

  status = nvjpegBufferPinnedCreate(nvjpeg_handle, NULL, &pinned_buffers[0]);
  TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create pinned buffer: ",
      status);

  status = nvjpegBufferPinnedCreate(nvjpeg_handle, NULL, &pinned_buffers[1]);
  TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create pinned buffer: ",
      status);

  status = nvjpegBufferDeviceCreate(nvjpeg_handle, NULL, &device_buffer);
  TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create device buffer: ",
      status);

  status = nvjpegJpegStreamCreate(nvjpeg_handle, &jpeg_streams[0]);
  TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create jpeg stream: ",
      status);

  status = nvjpegJpegStreamCreate(nvjpeg_handle, &jpeg_streams[1]);
  TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create jpeg stream: ",
      status);

  status = nvjpegDecodeParamsCreate(nvjpeg_handle, &nvjpeg_decode_params);
  TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create decode params: ",
      status);

  cudaStatus = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  TORCH_CHECK(
      cudaStatus == cudaSuccess, "Failed to create CUDA stream: ", cudaStatus);
}

CUDAJpegDecoder::~CUDAJpegDecoder() {
  nvjpegStatus_t status;
  cudaError_t cudaStatus;

  std::cout << "Destroying CUDAJpegDecoder 1" << std::endl;

  status = nvjpegDecodeParamsDestroy(nvjpeg_decode_params);
  TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to destroy nvjpeg decode params: ",
      status);

  std::cout << "Destroying CUDAJpegDecoder 2" << std::endl;

  status = nvjpegJpegStreamDestroy(jpeg_streams[0]);
  TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to destroy jpeg stream: ",
      status);

  std::cout << "Destroying CUDAJpegDecoder 3" << std::endl;

  status = nvjpegJpegStreamDestroy(jpeg_streams[1]);
  TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to destroy jpeg stream: ",
      status);

  std::cout << "Destroying CUDAJpegDecoder 4" << std::endl;

  status = nvjpegBufferPinnedDestroy(pinned_buffers[0]);
  TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to destroy pinned buffer[0]: ",
      status);
  std::cout << "Destroying CUDAJpegDecoder 5" << std::endl;
  status = nvjpegBufferPinnedDestroy(pinned_buffers[1]);
  TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to destroy pinned buffer[1]: ",
      status);
  std::cout << "Destroying CUDAJpegDecoder 6" << std::endl;
  status = nvjpegBufferDeviceDestroy(device_buffer);
  TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to destroy device buffer: ",
      status);
  std::cout << "Destroying CUDAJpegDecoder 7" << std::endl;
  status = nvjpegJpegStateDestroy(nvjpeg_decoupled_state);
  TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to destroy nvjpeg decoupled state: ",
      status);
  std::cout << "Destroying CUDAJpegDecoder 8" << std::endl;
  status = nvjpegDecoderDestroy(nvjpeg_decoder);
  TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to destroy nvjpeg decoder: ",
      status);
  std::cout << "Destroying CUDAJpegDecoder 9" << std::endl;
  status = nvjpegJpegStateDestroy(nvjpeg_state);
  TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to destroy nvjpeg state: ",
      status);

  cudaStatus = cudaStreamDestroy(stream);
  TORCH_CHECK(
      cudaStatus == cudaSuccess, "Failed to destroy CUDA stream: ", cudaStatus);

  nvjpegStatus_t destroy_status = nvjpegDestroy(nvjpeg_handle);
  TORCH_CHECK(
      destroy_status == NVJPEG_STATUS_SUCCESS,
      "nvjpegDestroy failed: ",
      destroy_status);
}

std::tuple<
    std::vector<nvjpegImage_t>,
    std::vector<torch::Tensor>,
    std::vector<int>>
CUDAJpegDecoder::prepare_buffers(
    const std::vector<torch::Tensor>& encoded_images,
    const nvjpegOutputFormat_t& output_format,
    const torch::Device& device) {
  /*
    This function scans the encoded images' jpeg headers and
    allocates decoding buffers based on the metadata found

    Args:
    - encoded_images (std::vector<torch::Tensor>): a vector of tensors
    containing the jpeg bitstreams to be decoded. Each tensor must have dtype
    torch.uint8 and device cpu
    - output_format (nvjpegOutputFormat_t): NVJPEG_OUTPUT_RGB, NVJPEG_OUTPUT_Y
    or NVJPEG_OUTPUT_UNCHANGED
    - device (torch::Device): The desired CUDA device for the returned Tensors

    Returns:
    - decoded_images (std::vector<nvjpegImage_t>): a vector of nvjpegImages
    containing pointers to the memory of the decoded images
    - output_tensors (std::vector<torch::Tensor>): a vector of Tensors
    containing the decoded images. `decoded_images` points to the memory of
    output_tensors
    - channels (std::vector<int>): a vector of ints containing the number of
    output image channels for every image
  */

  int width[NVJPEG_MAX_COMPONENT];
  int height[NVJPEG_MAX_COMPONENT];
  std::vector<int> channels(encoded_images.size());
  nvjpegChromaSubsampling_t subsampling;
  nvjpegStatus_t status;

  std::vector<torch::Tensor> output_tensors{encoded_images.size()};
  std::vector<nvjpegImage_t> decoded_images{encoded_images.size()};

  for (std::vector<at::Tensor>::size_type i = 0; i < encoded_images.size();
       i++) {
    // extract bitstream meta data to figure out the number of channels, height,
    // width for every image
    status = nvjpegGetImageInfo(
        nvjpeg_handle,
        (unsigned char*)encoded_images[i].data_ptr(),
        encoded_images[i].numel(),
        &channels[i],
        &subsampling,
        width,
        height);
    TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS, "Failed to get image info: ", status);

    TORCH_CHECK(
        subsampling != NVJPEG_CSS_UNKNOWN, "Unknown chroma subsampling");

    // output channels may be different from the actual number of channels in
    // the image, e.g. we decode a grayscale image as RGB and slice off the
    // extra channels later
    int output_channels = 3;
    if (output_format == NVJPEG_OUTPUT_RGB ||
        output_format == NVJPEG_OUTPUT_UNCHANGED) {
      output_channels = 3;
    } else if (output_format == NVJPEG_OUTPUT_Y) {
      output_channels = 1;
    }

    // reserve output buffer
    auto output_tensor = torch::empty(
        {int64_t(output_channels), int64_t(height[0]), int64_t(width[0])},
        torch::dtype(torch::kU8).device(device));
    output_tensors[i] = output_tensor;

    // fill nvjpegImage_t struct
    for (int c = 0; c < output_channels; c++) {
      decoded_images[i].channel[c] = output_tensor[c].data_ptr<uint8_t>();
      decoded_images[i].pitch[c] = width[0];
    }
    for (int c = output_channels; c < NVJPEG_MAX_COMPONENT; c++) {
      decoded_images[i].channel[c] = NULL;
      decoded_images[i].pitch[c] = 0;
    }
  }
  return {decoded_images, output_tensors, channels};
}

std::vector<torch::Tensor> CUDAJpegDecoder::decode_images(
    const std::vector<torch::Tensor>& encoded_images,
    const nvjpegOutputFormat_t& output_format,
    const torch::Device& device) {
  /*
    This function decodes a batch of jpeg bitstreams.
    We scan all encoded bitstreams and sort them into two groups:
    1. Baseline JPEGs: Can be decoded with hardware support on A100+ GPUs.
    2. Other JPEGs (e.g. progressive JPEGs): Need to be decoded in software.

    Args:
    - encoded_images (std::vector<torch::Tensor>): a vector of tensors
    containing the jpeg bitstreams to be decoded
    - output_format (nvjpegOutputFormat_t): NVJPEG_OUTPUT_RGB, NVJPEG_OUTPUT_Y
    or NVJPEG_OUTPUT_
    - device (torch::Device): The desired CUDA device for the returned Tensors

    Returns:
    - output_tensors (std::vector<torch::Tensor>): a vector of Tensors
    containing the decoded images
  */

  auto [decoded_imgs_buf, output_tensors, channels] =
      prepare_buffers(encoded_images, output_format, device);

  nvjpegStatus_t status;
  cudaError_t cudaStatus;

  cudaStatus = cudaStreamSynchronize(stream);
  TORCH_CHECK(
      cudaStatus == cudaSuccess,
      "Failed to synchronize CUDA stream: ",
      cudaStatus);

  // baseline JPEGs can be batch decoded with hardware support on A100+ GPUs
  // ultra fast!
  std::vector<const unsigned char*> hw_input_buffer;
  std::vector<size_t> hw_input_buffer_size;
  std::vector<nvjpegImage_t> hw_output_buffer;

  // other JPEG types such as progressive JPEGs can be decoded one-by-one in
  // software slow :(
  std::vector<const unsigned char*> sw_input_buffer;
  std::vector<size_t> sw_input_buffer_size;
  std::vector<nvjpegImage_t> sw_output_buffer;

  if (hw_decode_available) {
    for (std::vector<at::Tensor>::size_type i = 0; i < encoded_images.size();
         ++i) {
      // extract bitstream meta data to figure out whether a bit-stream can be
      // decoded
      nvjpegJpegStreamParseHeader(
          nvjpeg_handle,
          encoded_images[i].data_ptr<uint8_t>(),
          encoded_images[i].numel(),
          jpeg_streams[0]);
      int isSupported = -1;
      nvjpegDecodeBatchedSupported(
          nvjpeg_handle, jpeg_streams[0], &isSupported);

      if (isSupported == 0) {
        hw_input_buffer.push_back(encoded_images[i].data_ptr<uint8_t>());
        hw_input_buffer_size.push_back(encoded_images[i].numel());
        hw_output_buffer.push_back(decoded_imgs_buf[i]);
      } else {
        sw_input_buffer.push_back(encoded_images[i].data_ptr<uint8_t>());
        sw_input_buffer_size.push_back(encoded_images[i].numel());
        sw_output_buffer.push_back(decoded_imgs_buf[i]);
      }
    }
  } else {
    for (std::vector<at::Tensor>::size_type i = 0; i < encoded_images.size();
         ++i) {
      sw_input_buffer.push_back(encoded_images[i].data_ptr<uint8_t>());
      sw_input_buffer_size.push_back(encoded_images[i].numel());
      sw_output_buffer.push_back(decoded_imgs_buf[i]);
    }
  }

  if (hw_input_buffer.size() > 0) {
    // UNCHANGED behaves weird, so we use RGB instead
    status = nvjpegDecodeBatchedInitialize(
        nvjpeg_handle,
        nvjpeg_state,
        hw_input_buffer.size(),
        1,
        output_format == NVJPEG_OUTPUT_UNCHANGED ? NVJPEG_OUTPUT_RGB
                                                 : output_format);
    TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to initialize batch decoding: ",
        status);

    status = nvjpegDecodeBatched(
        nvjpeg_handle,
        nvjpeg_state,
        hw_input_buffer.data(),
        hw_input_buffer_size.data(),
        hw_output_buffer.data(),
        stream);
    TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS, "Failed to decode batch: ", status);
  }

  if (sw_input_buffer.size() > 0) {
    status =
        nvjpegStateAttachDeviceBuffer(nvjpeg_decoupled_state, device_buffer);
    TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to attach device buffer: ",
        status);
    int buffer_index = 0;
    // UNCHANGED behaves weird, so we use RGB instead
    status = nvjpegDecodeParamsSetOutputFormat(
        nvjpeg_decode_params,
        output_format == NVJPEG_OUTPUT_UNCHANGED ? NVJPEG_OUTPUT_RGB
                                                 : output_format);
    TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to set output format: ",
        status);
    for (std::vector<at::Tensor>::size_type i = 0; i < sw_input_buffer.size();
         ++i) {
      status = nvjpegJpegStreamParse(
          nvjpeg_handle,
          sw_input_buffer[i],
          sw_input_buffer_size[i],
          0,
          0,
          jpeg_streams[buffer_index]);
      TORCH_CHECK(
          status == NVJPEG_STATUS_SUCCESS,
          "Failed to parse jpeg stream: ",
          status);

      status = nvjpegStateAttachPinnedBuffer(
          nvjpeg_decoupled_state, pinned_buffers[buffer_index]);
      TORCH_CHECK(
          status == NVJPEG_STATUS_SUCCESS,
          "Failed to attach pinned buffer: ",
          status);

      status = nvjpegDecodeJpegHost(
          nvjpeg_handle,
          nvjpeg_decoder,
          nvjpeg_decoupled_state,
          nvjpeg_decode_params,
          jpeg_streams[buffer_index]);
      TORCH_CHECK(
          status == NVJPEG_STATUS_SUCCESS,
          "Failed to decode jpeg stream: ",
          status);

      cudaStatus = cudaStreamSynchronize(stream);
      TORCH_CHECK(
          cudaStatus == cudaSuccess,
          "Failed to synchronize CUDA stream: ",
          cudaStatus);

      status = nvjpegDecodeJpegTransferToDevice(
          nvjpeg_handle,
          nvjpeg_decoder,
          nvjpeg_decoupled_state,
          jpeg_streams[buffer_index],
          stream);
      TORCH_CHECK(
          status == NVJPEG_STATUS_SUCCESS,
          "Failed to transfer jpeg to device: ",
          status);

      buffer_index = 1 - buffer_index; // switch pinned buffer in pipeline mode
                                       // to avoid an extra sync

      status = nvjpegDecodeJpegDevice(
          nvjpeg_handle,
          nvjpeg_decoder,
          nvjpeg_decoupled_state,
          &sw_output_buffer[i],
          stream);
      TORCH_CHECK(
          status == NVJPEG_STATUS_SUCCESS,
          "Failed to decode jpeg stream: ",
          status);
    }
  }

  cudaStatus = cudaStreamSynchronize(stream);
  TORCH_CHECK(
      cudaStatus == cudaSuccess,
      "Failed to synchronize CUDA stream: ",
      cudaStatus);

  // prune extraneous channels from single channel images
  if (output_format == NVJPEG_OUTPUT_UNCHANGED)
    for (std::vector<at::Tensor>::size_type i = 0; i < output_tensors.size();
         ++i)
      if (channels[i] == 1)
        output_tensors[i] = output_tensors[i][0].unsqueeze(0).clone();

  return output_tensors;
}

} // namespace image
} // namespace vision

#endif

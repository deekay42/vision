#include "encode_jpegs_cuda.h"
#if !NVJPEG_FOUND
std::vector<torch::Tensor> encode_jpegs_cuda(
    const std::vector<torch::Tensor>& images,
    const int64_t quality) {
  TORCH_CHECK(
      false, "encode_jpegs_cuda: torchvision not compiled with nvJPEG support");
}
#else

#include <torch/nn/functional.h>
#include "c10/core/ScalarType.h"
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <iostream>
#include <memory>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>


namespace vision {
namespace image {

// We use global variables to cache the encoder and decoder instances and
// reuse them across calls to the corresponding pytorch functions
// Lastly, we only allow single threaded access for now so we lock the mutex
std::mutex encoderMutex;
std::unique_ptr<CUDAJpegEncoder> cudaJpegEncoder;


std::vector<torch::Tensor> encode_jpegs_cuda(
    const std::vector<torch::Tensor>& decoded_images,
    const int64_t quality)
    {
      C10_LOG_API_USAGE_ONCE(
      "torchvision.csrc.io.image.cuda.nvjpeg.encode_jpegs_cuda");

  // some nvjpeg structures are not thread safe so we're keeping it single
  // threaded for now. In the future this may be an opportunity to unlock
  // further speedups
      std::lock_guard<std::mutex> lock(encoderMutex);
      TORCH_CHECK(decoded_images.size() > 0, "Empty input tensor list");
      torch::Device device = decoded_images[0].device();


      // lazy init of the encoder class
      // the encoder holds on to a lot of state and is expensive to create, so we
      // reuse it across calls.
      // NB: the cached structures are device specific and cannot be reused across devices
      if (cudaJpegEncoder == nullptr || device != cudaJpegEncoder->device) {
        std::cout << "Building new CUDAJpegEncoder instance. This may take a while... <3" << std::endl;
        if (cudaJpegEncoder != nullptr)
        {
          std::cout << "it's a new device!" << std::endl;
          delete cudaJpegEncoder.release();
        }
        cudaSetDevice(device.index());
        cudaJpegEncoder = std::make_unique<CUDAJpegEncoder>(device);
        // Unfortunately, we cannot rely on the smart pointer releasing the encoder object correctly upon program exit.
        // This is because, when cudaJpegEncoder gets destroyed, the CUDA runtime may already be shut down,
        // rendering all destroy* calls in the encoder destructor invalid.
        // Instead, we use an atexit hook which executes after main() finishes, but before CUDA shuts down when the program exits.
        std::atexit([](){delete cudaJpegEncoder.release();});
      }

  std::vector<torch::Tensor> contig_images;
  contig_images.reserve(decoded_images.size());
  for (const auto& image : decoded_images) {
    TORCH_CHECK(
        image.dtype() == torch::kU8, "Input tensor dtype should be uint8");

    TORCH_CHECK(
        image.device() == device,
        "All input tensors must be on the same CUDA device when encoding with nvjpeg")

    TORCH_CHECK(
        image.dim() == 3 && image.numel() > 0,
        "Input data should be a 3-dimensional tensor");

    TORCH_CHECK(
        image.size(0) == 3,
        "The number of channels should be 3, got: ",
        image.size(0));

    // nvjpeg requires images to be contiguous
    if (image.is_contiguous()) {
      contig_images.push_back(image);
    } else {
      contig_images.push_back(image.contiguous());
    }
  }

    cudaJpegEncoder->setQuality(quality);
    std::vector<torch::Tensor> encoded_images;
    for (const auto& image : contig_images) {
      auto encoded_image = cudaJpegEncoder->encode_jpeg(image, device);
      encoded_images.push_back(encoded_image);
    }
    return encoded_images;

    }

CUDAJpegEncoder::CUDAJpegEncoder(const torch::Device& device) : device{device} {

    nvjpegStatus_t status;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(device.index());
    TORCH_CHECK(
      cudaStatus == cudaSuccess, "Failed to set CUDA device: ", cudaStatus);


    status = nvjpegCreateSimple(&nvjpeg_handle);
    TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to create nvjpeg handle: ",
        status);

    cudaStatus = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    TORCH_CHECK(
      cudaStatus == cudaSuccess, "Failed to create CUDA stream: ", cudaStatus);

    status =
        nvjpegEncoderStateCreate(nvjpeg_handle, &nv_enc_state, stream);
    TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to create nvjpeg encoder state: ",
        status);

    status =
        nvjpegEncoderParamsCreate(nvjpeg_handle, &nv_enc_params, stream);
    TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to create nvjpeg encoder params: ",
        status);
}

CUDAJpegEncoder::~CUDAJpegEncoder() {
    nvjpegStatus_t status;
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(device.index());
    TORCH_CHECK(
      cudaStatus == cudaSuccess, "Failed to set CUDA device: ", cudaStatus);

    status = nvjpegEncoderParamsDestroy(nv_enc_params);
    TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to destroy nvjpeg encoder params: ",
        status);

    status = nvjpegEncoderStateDestroy(nv_enc_state);
    TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to destroy nvjpeg encoder state: ",
        status);

    cudaStreamSynchronize(stream);
    cudaStatus = cudaStreamDestroy(stream);
    TORCH_CHECK(
      cudaStatus == cudaSuccess, "Failed to destroy CUDA stream: ", cudaStatus);

    status = nvjpegDestroy(nvjpeg_handle);
    TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "nvjpegDestroy failed: ",
        status);
}

int getDeviceForStream(cudaStream_t stream) {
    // Temporary storage for the stream flags
    unsigned int flags;
    // Save the current device so we can restore it later
    int originalDevice;
    cudaGetDevice(&originalDevice);
    // This function call sets the device to the one associated with the stream
    cudaError_t result = cudaStreamGetFlags(stream, &flags);
    if (result != cudaSuccess) {
        std::cerr << "Failed to get stream flags: " << cudaGetErrorString(result) << std::endl;
        return -1; // Return -1 on failure
    }
    // Get the current device, which should now be the one associated with the stream
    int currentDevice;
    cudaGetDevice(&currentDevice);
    // Restore the original device
    cudaSetDevice(originalDevice);
    return currentDevice;
}

torch::Tensor CUDAJpegEncoder::encode_jpeg(
    const torch::Tensor& src_image,
    const torch::Device& device) {

  cudaError_t cudaStatus = cudaSetDevice(device.index());
    TORCH_CHECK(
      cudaStatus == cudaSuccess, "Failed to set CUDA device: ", cudaStatus);

  int channels = src_image.size(0);
  int height = src_image.size(1);
  int width = src_image.size(2);

  nvjpegStatus_t samplingSetResult = nvjpegEncoderParamsSetSamplingFactors(
      nv_enc_params, NVJPEG_CSS_444, stream);
  TORCH_CHECK(
      samplingSetResult == NVJPEG_STATUS_SUCCESS,
      "Failed to set nvjpeg encoder params sampling factors: ",
      samplingSetResult);
  std::cout << "1" << std::endl;
  // Create nvjpeg image
  nvjpegImage_t target_image;
  for (int c = 0; c < channels; c++) {
    target_image.channel[c] = src_image[c].data_ptr<uint8_t>();
    // this is why we need contiguous tensors
    target_image.pitch[c] = width;
  }
  for (int c = channels; c < NVJPEG_MAX_COMPONENT; c++) {
    target_image.channel[c] = nullptr;
    target_image.pitch[c] = 0;
  }
  nvjpegStatus_t encodingState;
  std::cout << "2" << std::endl;
  // Encode the image
  encodingState = nvjpegEncodeImage(
      nvjpeg_handle,
      nv_enc_state,
      nv_enc_params,
      &target_image,
      NVJPEG_INPUT_RGB,
      width,
      height,
      stream);

  TORCH_CHECK(
      encodingState == NVJPEG_STATUS_SUCCESS,
      "image encoding failed: ",
      encodingState);
  std::cout << "3" << std::endl;
  // Retrieve length of the encoded image
  size_t length;
  nvjpegStatus_t getStreamState = nvjpegEncodeRetrieveBitstreamDevice(
      nvjpeg_handle, nv_enc_state, NULL, &length, stream);
  TORCH_CHECK(
      getStreamState == NVJPEG_STATUS_SUCCESS,
      "Failed to retrieve encoded image stream state: ",
      getStreamState);

  // Synchronize the stream to ensure that the encoded image is ready
  cudaError_t syncState = cudaStreamSynchronize(stream);
  TORCH_CHECK(syncState == cudaSuccess, "CUDA ERROR: ", syncState);

  std::cout << "size of original is: " << src_image.sizes() << std::endl;
  std::cout << "retrieved size is: " << length << std::endl;
  std::cout << "stream device: " << getDeviceForStream(stream) << std::endl;
    std::cout << "4" << std::endl;



  getStreamState = nvjpegEncodeRetrieveBitstreamDevice(
      nvjpeg_handle, nv_enc_state, NULL, &length, stream);
  TORCH_CHECK(
      getStreamState == NVJPEG_STATUS_SUCCESS,
      "Failed to retrieve encoded image stream state: ",
      getStreamState);

  // Synchronize the stream to ensure that the encoded image is ready
  syncState = cudaStreamSynchronize(stream);
  TORCH_CHECK(syncState == cudaSuccess, "CUDA ERROR: ", syncState);

  std::cout << "1size of original is: " << src_image.sizes() << std::endl;
  std::cout << "1retrieved size is: " << length << std::endl;
  std::cout << "1stream device: " << getDeviceForStream(stream) << std::endl;
    std::cout << "14" << std::endl;

  // Reserve buffer for the encoded image
  torch::Tensor encoded_image = torch::empty(
      {static_cast<long>(length)},
      torch::TensorOptions()
          .dtype(torch::kByte)
          .layout(torch::kStrided)
          .device(device)
          .requires_grad(false));
  syncState = cudaStreamSynchronize(stream);
  TORCH_CHECK(syncState == cudaSuccess, "CUDA ERROR: ", syncState);
std::cout << "5" << std::endl;
  // Retrieve the encoded image
  getStreamState = nvjpegEncodeRetrieveBitstreamDevice(
      nvjpeg_handle,
      nv_enc_state,
      encoded_image.data_ptr<uint8_t>(),
      &length,
      0);
  TORCH_CHECK(
      getStreamState == NVJPEG_STATUS_SUCCESS,
      "Failed to retrieve encoded image: ",
      getStreamState);
      std::cout << "6" << std::endl;
  return encoded_image;
}

void CUDAJpegEncoder::setQuality(const int64_t quality)
  {
    nvjpegStatus_t paramsQualityStatus =
        nvjpegEncoderParamsSetQuality(nv_enc_params, quality, stream);
    TORCH_CHECK(
        paramsQualityStatus == NVJPEG_STATUS_SUCCESS,
        "Failed to set nvjpeg encoder params quality: ",
        paramsQualityStatus);
  }

} // namespace image
} // namespace vision

#endif // NVJPEG_FOUND

#if !defined(C10_MOBILE) && !defined(ANDROID)

#include <c10/util/error.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>

#include <fmt/format.h>
#include <miniz.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

namespace {

std::string create_temp_dir() {
#ifdef _WIN32
  throw std::runtime_error("Not implemented");
#else
  std::string temp_dir = "/tmp/XXXXXX";
  if (mkdtemp(temp_dir.data()) == nullptr) {
    throw std::runtime_error(
        std::string("Failed to create temporary directory: ") +
        c10::utils::str_error(errno));
  }
  return temp_dir;
#endif
}
} // namespace

namespace torch::inductor {

namespace {
const nlohmann::json& load_json_file(const std::filesystem::path& json_path) {
  if (!std::filesystem::exists(json_path)) {
    throw std::runtime_error("File not found: " + json_path.string());
  }

  std::ifstream json_file(json_path);
  TORCH_CHECK(json_file.is_open());
  static nlohmann::json json_obj;
  json_file >> json_obj;

  return json_obj;
}

std::tuple<std::string, std::string> get_cpp_compile_command(
    const std::string& filename,
    const std::vector<std::string>& sources,
    const nlohmann::json& compile_options,
    const std::string& output_dir = "") {
  // Construct the cpp command

  std::string compiler = compile_options["compiler"].get<std::string>();
  bool compile_only = compile_options["compile_only"].get<bool>();

  std::string source_args;
  for (const std::string& source : sources) {
    source_args += source + " ";
  }

  std::string file_ext = compile_only ? ".o" : ".so";
  std::string target_file = output_dir + filename + file_ext;
  std::filesystem::path target_dir = output_dir;
  if (target_dir.empty()) {
    target_dir = std::filesystem::path(filename).parent_path();
  }

  std::string cflags_args;
  for (auto& arg : compile_options["cflags"]) {
    cflags_args += "-" + arg.get<std::string>() + " ";
  }

  std::string definitions_args;
  for (auto& arg : compile_options["definitions"]) {
    definitions_args += "-D " + arg.get<std::string>() + " ";
  }

  std::string include_dirs_args;
  for (auto& arg : compile_options["include_dirs"]) {
    include_dirs_args += "-I" + arg.get<std::string>() + " ";
  }

  std::string ldflags_args;
  for (auto& arg : compile_options["ldflags"]) {
    ldflags_args += "-" + arg.get<std::string>() + " ";
  }

  std::string libraries_dirs_args;
  for (auto& arg : compile_options["libraries_dirs"]) {
    libraries_dirs_args += "-L" + arg.get<std::string>() + " ";
  }

  std::string libraries_args;
  for (auto& arg : compile_options["libraries"]) {
    libraries_args += "-l" + arg.get<std::string>() + " ";
  }

  std::string passthrough_parameters_args;
  std::string target = "script.ld";
  auto replacement = (target_dir / target).string();
  for (auto& arg : compile_options["passthrough_args"]) {
    std::string arg_str = arg.get<std::string>();
    size_t pos = arg_str.find(target);
    if (pos != std::string::npos) {
      arg_str.replace(pos, target.length(), replacement);
    }
    passthrough_parameters_args += arg_str + " ";
  }

  std::string compile_only_arg = compile_only ? "-c" : "";

  std::string cmd = fmt::format(
      "{} {} {} {} {} {} {} {} {} {} -o {}",
      compiler,
      source_args,
      definitions_args,
      cflags_args,
      include_dirs_args,
      passthrough_parameters_args,
      ldflags_args,
      libraries_args,
      libraries_dirs_args,
      compile_only_arg,
      target_file);

  return std::make_tuple(cmd, target_file);
}

std::string compile_so(
    const std::string& cpp_filename,
    const std::string& consts_filename) {
  // Compile the cpp file into a .so

  size_t lastindex = cpp_filename.find_last_of('.');
  std::string filename = cpp_filename.substr(0, lastindex);

  std::string compile_flags_path = filename + "_compile_flags.json";
  const nlohmann::json compile_flags = load_json_file(compile_flags_path);

  auto [compile_cmd, output_o] =
      get_cpp_compile_command(filename, {cpp_filename}, compile_flags);

  std::string linker_flags_path =
      cpp_filename.substr(0, lastindex) + "_linker_flags.json";
  const nlohmann::json linker_flags = load_json_file(linker_flags_path);

  auto [link_cmd, output_so] = get_cpp_compile_command(
      filename, {output_o, consts_filename}, linker_flags);

  // Run the commands to generate a .so file
  int status = system(compile_cmd.c_str());
  if (status != 0) {
    throw std::runtime_error("Failed to compile cpp file.");
  }
  status = system(link_cmd.c_str());
  if (status != 0) {
    throw std::runtime_error("Failed to link files.");
  }

  // Move the mmapped weights onto the .so
  std::string serialized_weights_path = filename + "_serialized_weights.bin";
  if (std::filesystem::exists(serialized_weights_path)) {
    std::ifstream serialized_weights_file(
        serialized_weights_path, std::ios::binary);
    if (!serialized_weights_file.is_open()) {
      throw std::runtime_error("Failed to open serialized weights file");
    }
    std::vector<char> serialized_weights(
        (std::istreambuf_iterator<char>(serialized_weights_file)),
        std::istreambuf_iterator<char>());
    serialized_weights_file.close();

    std::ofstream output_so_file(output_so, std::ios::binary | std::ios::app);
    if (!output_so_file.is_open()) {
      throw std::runtime_error("Failed to open output .so file");
    }
    // Page align the weights
    std::streampos so_size = output_so_file.tellp();
    std::vector<char> padding(16384 - so_size % 16384, ' ');
    output_so_file.write(
        padding.data(), static_cast<std::streamsize>(padding.size()));
    output_so_file.write(
        serialized_weights.data(),
        static_cast<std::streamsize>(serialized_weights.size()));
    output_so_file.close();
  }

  return output_so;
}
} // namespace

void AOTIModelPackageLoader::load_metadata(const std::string& cpp_filename) {
  // Parse metadata json file (if it exists) into the metadata_ map
  size_t lastindex = cpp_filename.find_last_of('.');
  std::string metadata_json_path =
      cpp_filename.substr(0, lastindex) + "_metadata.json";

  const nlohmann::json metadata_json_obj = load_json_file(metadata_json_path);

  for (auto& item : metadata_json_obj.items()) {
    metadata_[item.key()] = item.value().get<std::string>();
  }
}

AOTIModelPackageLoader::AOTIModelPackageLoader(
    const std::string& model_package_path,
    const std::string& model_name,
    const bool run_single_threaded,
    const size_t num_runners) {
  if (run_single_threaded) {
    if (num_runners != 1) {
      throw std::runtime_error(
          "num_runners must be 1 when run_single_threaded is true");
    }
  } else {
    if (num_runners < 1) {
      throw std::runtime_error(
          "num_runners must be >=1 when run_single_threaded is false");
    }
  }

  // Extract all files within the zipfile to a temporary directory
  mz_zip_archive zip_archive;
  memset(&zip_archive, 0, sizeof(zip_archive));

  if (!mz_zip_reader_init_file(&zip_archive, model_package_path.c_str(), 0)) {
    throw std::runtime_error(
        std::string("Failed to initialize zip archive: ") +
        mz_zip_get_error_string(mz_zip_get_last_error(&zip_archive)));
  }

  temp_dir_ = create_temp_dir();
  std::string so_filename;
  std::string cpp_filename;
  std::string consts_filename;
  std::string found_filenames; // Saving for bookkeeping
  auto model_directory =
      std::filesystem::path("data") / "aotinductor" / model_name;
  auto const_directory = std::filesystem::path("data") / "constants";

  for (uint32_t i = 0; i < zip_archive.m_total_files; i++) {
    uint32_t filename_len =
        mz_zip_reader_get_filename(&zip_archive, i, nullptr, 0);
    if (filename_len == 0) {
      throw std::runtime_error("Failed to read filename");
    }
    // filename_len returned by mz_zip_reader_get_filename includes the null
    // terminator, so we need to subtract 1 here
    std::string filename_str(filename_len - 1, '\0');
    if (!mz_zip_reader_get_filename(
            &zip_archive, i, filename_str.data(), filename_len)) {
      throw std::runtime_error("Failed to read filename");
    }
    std::filesystem::path filename(filename_str.c_str());

    found_filenames += filename_str;
    found_filenames += " ";

    // Only compile files in the specified model directory
    bool in_model_directory =
        (!std::filesystem::relative(filename, model_directory).empty());
    bool in_const_directory =
        (!std::filesystem::relative(filename, const_directory).empty());
    if (in_model_directory || in_const_directory) {
      std::filesystem::path output_path = temp_dir_;

      if (in_model_directory) {
        output_path /= filename;
      } else {
        // Extract constants to the same directory as the rest of the files
        // to be consistent with internal implementation
        output_path /= model_directory;
        output_path /= filename.filename();
      }
      auto output_path_str = output_path.string();

      LOG(INFO) << "Extract file: " << filename_str << " to " << output_path;

      // Create the parent directory if it doesn't exist
      if (!output_path.has_parent_path()) {
        throw std::runtime_error(
            "Failed to find parent path in " + output_path_str);
      }
      auto parent_path = output_path.parent_path();
      std::error_code ec{};
      std::filesystem::create_directories(parent_path, ec);
      if (!std::filesystem::is_directory(parent_path)) {
        throw std::runtime_error(fmt::format(
            "Failed to create directory {}: {}",
            parent_path.string(),
            ec.message()));
      }

      // Extracts file to the temp directory
      mz_zip_reader_extract_file_to_file(
          &zip_archive, filename_str.c_str(), output_path_str.c_str(), 0);

      // Save the file for bookkeeping
      if (output_path.has_extension()) {
        auto filename_extension = output_path.extension();
        if (filename_extension == ".cpp") {
          cpp_filename = output_path_str;
        } else if (filename_extension == ".o") {
          consts_filename = output_path_str;
        } else if (filename_extension == ".so") {
          so_filename = output_path_str;
        }
      }
    }
  }

  // Close the zip archive as we have extracted all files to the temp
  // directory
  if (!mz_zip_reader_end(&zip_archive)) {
    throw std::runtime_error(
        std::string("Failed to close zip archive: {}") +
        mz_zip_get_error_string(mz_zip_get_last_error(&zip_archive)));
  }

  if (cpp_filename.empty() && so_filename.empty()) {
    throw std::runtime_error(
        "No AOTInductor generate cpp file or so file found in zip archive. Loaded the following:\n" +
        found_filenames);
  }

  // Compile the .so
  std::string so_path = !so_filename.empty()
      ? so_filename
      : compile_so(cpp_filename, consts_filename);

  // Load metadata which can be queried by user
  load_metadata(cpp_filename);

  // Construct the runner depending on the device information
  std::string device = metadata_["AOTI_DEVICE_KEY"];

  if (device.empty()) {
    throw std::runtime_error("No device information found.");
  }

  std::unordered_map<std::string, CreateAOTIModelRunnerFunc>
      registered_aoti_runner = getAOTIModelRunnerRegistry();

  if (registered_aoti_runner.find(device) == registered_aoti_runner.end()) {
    throw std::runtime_error("Unsupported device found: " + device);
  }

  std::string cubin_dir = (temp_dir_ / model_directory).string();
  runner_ = registered_aoti_runner[device](
      so_path, num_runners, device, cubin_dir, run_single_threaded);
}

AOTIModelPackageLoader::~AOTIModelPackageLoader() {
  // Clean up the temporary directory
  if (!temp_dir_.empty()) {
    std::error_code ec{};
    // The noexcept version of remove_all is used
    std::filesystem::remove_all(temp_dir_, ec);
  }
}

AOTIModelContainerRunner* AOTIModelPackageLoader::get_runner() {
  return runner_.get();
}

std::vector<at::Tensor> AOTIModelPackageLoader::run(
    const std::vector<at::Tensor>& inputs,
    void* stream_handle) {
  return runner_->run(inputs, stream_handle);
}

std::vector<at::Tensor> AOTIModelPackageLoader::boxed_run(
    std::vector<at::Tensor>&& inputs,
    void* stream_handle) {
  return runner_->boxed_run(std::move(inputs), stream_handle);
}

std::unordered_map<std::string, std::string> AOTIModelPackageLoader::
    get_metadata() {
  return metadata_;
}

std::vector<std::string> AOTIModelPackageLoader::get_call_spec() {
  return runner_->get_call_spec();
}

void AOTIModelPackageLoader::load_constants(
    std::unordered_map<std::string, at::Tensor>& constants_map,
    bool use_inactive,
    bool check_full_update,
    bool user_managed) {
  std::unordered_map<std::string, std::string> constant_name_to_fqn =
      runner_->getConstantNamesToOriginalFQNs();
  std::unordered_map<std::string, std::string> fqn_to_constant_name;
  for (const auto& it : constant_name_to_fqn) {
    fqn_to_constant_name.emplace(it.second, it.first);
  }

  std::unordered_map<std::string, at::Tensor> updated_constants_map;
  for (const auto& it : constants_map) {
    if (fqn_to_constant_name.find(it.first) != fqn_to_constant_name.end()) {
      updated_constants_map.emplace(fqn_to_constant_name[it.first], it.second);
    } else {
      throw std::runtime_error("Constant not found: " + it.first);
    }
  }

  return runner_->update_constant_buffer(
      updated_constants_map, use_inactive, check_full_update, user_managed);
}

std::vector<std::string> AOTIModelPackageLoader::get_constant_fqns() {
  std::unordered_map<std::string, std::string> constant_name_to_fqn =
      runner_->getConstantNamesToOriginalFQNs();
  std::vector<std::string> constant_fqns;
  constant_fqns.reserve(constant_name_to_fqn.size());
  for (const auto& it : constant_name_to_fqn) {
    constant_fqns.push_back(it.second);
  }
  return constant_fqns;
}

void AOTIModelPackageLoader::update_constant_buffer(
    std::unordered_map<std::string, at::Tensor>& tensor_map,
    bool use_inactive,
    bool validate_full_updates,
    bool user_managed) {
  runner_->update_constant_buffer(
      tensor_map, use_inactive, validate_full_updates, user_managed);
}
} // namespace torch::inductor
#endif

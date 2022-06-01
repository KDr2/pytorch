# @lint-ignore-every BUCKRESTRICTEDSYNTAX

# NOTE: This file is shared by internal and OSS BUCK build.
# These load paths point to different files in internal and OSS environment
load("//tools/build_defs:fb_native_wrapper.bzl", "fb_native")
load("//tools/build_defs:glob_defs.bzl", "subdir_glob")
load("//tools/build_defs:fb_xplat_cxx_library.bzl", "fb_xplat_cxx_library")
load("//tools/build_defs:fb_xplat_genrule.bzl", "fb_xplat_genrule")


# common buck configs
def get_static_dispatch_backend():
    static_dispatch_backend = native.read_config("pt", "static_dispatch_backend", None)
    if static_dispatch_backend == None:
        return []
    return static_dispatch_backend.split(";")

# common consntants


# common functions
def get_aten_static_dispatch_backend_headers(existing_headers):
    static_backends = get_static_dispatch_backend()
    for backend in static_backends:
        if backend != "CPU":
            existing_headers["{}Functions.h".format(backend)] = ":gen_aten[{}Functions.h]".format(backend)
            existing_headers["{}Functions_inl.h".format(backend)] = ":gen_aten[{}Functions_inl.h]".format(backend)
    return existing_headers


# these targets are shared by internal and OSS BUCK
def define_buck_targets(
        buck_root,
        jit_core_headers,
        feature = None,
        labels = []):
    fb_xplat_cxx_library(
        name = "th_header",
        header_namespace = "",
        exported_headers = subdir_glob([
            # TH
            ("aten/src", "TH/*.h"),
            ("aten/src", "TH/*.hpp"),
            ("aten/src", "TH/generic/*.h"),
            ("aten/src", "TH/generic/*.hpp"),
            ("aten/src", "TH/generic/simd/*.h"),
            ("aten/src", "TH/vector/*.h"),
            ("aten/src", "TH/generic/*.c"),
            ("aten/src", "TH/generic/*.cpp"),
            ("aten/src/TH", "*.h"),  # for #include <THGenerateFloatTypes.h>
            # THNN
            ("aten/src", "THNN/*.h"),
            ("aten/src", "THNN/generic/*.h"),
            ("aten/src", "THNN/generic/*.c"),
        ]),
        feature = feature,
        labels = labels,
    )

    fb_xplat_cxx_library(
        name = "aten_header",
        header_namespace = "",
        exported_headers = subdir_glob([
            # ATen Core
            ("aten/src", "ATen/core/**/*.h"),
            ("aten/src", "ATen/ops/*.h"),
            # ATen Base
            ("aten/src", "ATen/*.h"),
            ("aten/src", "ATen/cpu/**/*.h"),
            ("aten/src", "ATen/detail/*.h"),
            ("aten/src", "ATen/quantized/*.h"),
            ("aten/src", "ATen/vulkan/*.h"),
            ("aten/src", "ATen/metal/*.h"),
            ("aten/src", "ATen/nnapi/*.h"),
            # ATen Native
            ("aten/src", "ATen/native/*.h"),
            ("aten/src", "ATen/native/ao_sparse/quantized/cpu/*.h"),
            ("aten/src", "ATen/native/cpu/**/*.h"),
            ("aten/src", "ATen/native/sparse/*.h"),
            ("aten/src", "ATen/native/nested/*.h"),
            ("aten/src", "ATen/native/quantized/*.h"),
            ("aten/src", "ATen/native/quantized/cpu/*.h"),
            ("aten/src", "ATen/native/transformers/*.h"),
            ("aten/src", "ATen/native/ufunc/*.h"),
            ("aten/src", "ATen/native/utils/*.h"),
            ("aten/src", "ATen/native/vulkan/ops/*.h"),
            ("aten/src", "ATen/native/xnnpack/*.h"),
            ("aten/src", "ATen/mps/*.h"),
            ("aten/src", "ATen/native/mps/*.h"),
            # Remove the following after modifying codegen for mobile.
            ("aten/src", "ATen/mkl/*.h"),
            ("aten/src", "ATen/native/mkl/*.h"),
            ("aten/src", "ATen/native/mkldnn/*.h"),
        ]),
        visibility = ["PUBLIC"],
        feature = feature,
        labels = labels,
    )

    fb_xplat_cxx_library(
        name = "aten_vulkan_header",
        header_namespace = "",
        exported_headers = subdir_glob([
            ("aten/src", "ATen/native/vulkan/*.h"),
            ("aten/src", "ATen/native/vulkan/api/*.h"),
            ("aten/src", "ATen/native/vulkan/ops/*.h"),
            ("aten/src", "ATen/vulkan/*.h"),
        ]),
        feature = feature,
        labels = labels,
        visibility = ["PUBLIC"],
    )

    fb_xplat_cxx_library(
        name = "jit_core_headers",
        header_namespace = "",
        exported_headers = subdir_glob([("", x) for x in jit_core_headers]),
        feature = feature,
        labels = labels,
    )

    fb_xplat_cxx_library(
        name = "torch_headers",
        header_namespace = "",
        exported_headers = subdir_glob(
            [
                ("torch/csrc/api/include", "torch/**/*.h"),
                ("", "torch/csrc/**/*.h"),
                ("", "torch/csrc/generic/*.cpp"),
                ("", "torch/script.h"),
                ("", "torch/library.h"),
                ("", "torch/custom_class.h"),
                ("", "torch/custom_class_detail.h"),
                # Add again due to namespace difference from aten_header.
                ("", "aten/src/ATen/*.h"),
                ("", "aten/src/ATen/quantized/*.h"),
            ],
            exclude = [
                # Don't need on mobile.
                "torch/csrc/Exceptions.h",
                "torch/csrc/python_headers.h",
                "torch/csrc/utils/auto_gil.h",
                "torch/csrc/jit/serialization/mobile_bytecode_generated.h",
            ],
        ),
        feature = feature,
        labels = labels,
        visibility = ["PUBLIC"],
        deps = [
            ":generated-version-header",
        ],
    )

    fb_xplat_cxx_library(
        name = "aten_test_header",
        header_namespace = "",
        exported_headers = subdir_glob([
            ("aten/src", "ATen/test/*.h"),
        ]),
    )

    fb_xplat_cxx_library(
        name = "torch_mobile_headers",
        header_namespace = "",
        exported_headers = subdir_glob(
            [
                ("", "torch/csrc/jit/mobile/*.h"),
            ],
        ),
        feature = feature,
        labels = labels,
        visibility = ["PUBLIC"],
    )

    fb_xplat_cxx_library(
        name = "generated_aten_config_header",
        header_namespace = "ATen",
        exported_headers = {
            "Config.h": ":generate_aten_config[Config.h]",
        },
        feature = feature,
        labels = labels,
    )

    fb_xplat_cxx_library(
        name = "generated-autograd-headers",
        header_namespace = "torch/csrc/autograd/generated",
        exported_headers = {
            "Functions.h": ":gen_aten_libtorch[autograd/generated/Functions.h]",
            "VariableType.h": ":gen_aten_libtorch[autograd/generated/VariableType.h]",
            "variable_factories.h": ":gen_aten_libtorch[autograd/generated/variable_factories.h]",
            # Don't build python bindings on mobile.
            #"python_functions.h",
        },
        feature = feature,
        labels = labels,
        visibility = ["PUBLIC"],
    )

    fb_xplat_cxx_library(
        name = "generated-version-header",
        header_namespace = "torch",
        exported_headers = {
            "version.h": ":generate-version-header[version.h]",
        },
        feature = feature,
        labels = labels,
    )

    fb_native.genrule(
        name = "generate-version-header",
        srcs = [
            "torch/csrc/api/include/torch/version.h.in",
            "version.txt",
        ],
        cmd = "$(exe {}tools/setup_helpers:gen-version-header) ".format(buck_root) + " ".join([
            "--template-path",
            "torch/csrc/api/include/torch/version.h.in",
            "--version-path",
            "version.txt",
            "--output-path",
            "$OUT/version.h",
        ]),
        outs = {
            "version.h": ["version.h"],
        },
        default_outs = ["."],
    )

    fb_native.genrule(
        name = "generate_aten_config",
        srcs = [
            "aten/src/ATen/Config.h.in",
        ],
        cmd = " ".join([
            "sed",
            "-e 's/@AT_MKLDNN_ENABLED@/ATEN_MKLDNN_ENABLED_FBXPLAT/g'",
            "-e 's/@AT_MKL_ENABLED@/ATEN_MKL_ENABLED_FBXPLAT/g'",
            "-e 's/@AT_MKL_SEQUENTIAL@/ATEN_MKL_SEQUENTIAL_FBXPLAT/g'",
            "-e 's/@AT_FFTW_ENABLED@/0/g'",
            "-e 's/@AT_POCKETFFT_ENABLED@/0/g'",
            "-e 's/@AT_NNPACK_ENABLED@/ATEN_NNPACK_ENABLED_FBXPLAT/g'",
            "-e 's/@CAFFE2_STATIC_LINK_CUDA_INT@/CAFFE2_STATIC_LINK_CUDA_FBXPLAT/g'",
            "-e 's/@AT_BUILD_WITH_BLAS@/USE_BLAS_FBXPLAT/g'",
            "-e 's/@AT_PARALLEL_OPENMP@/AT_PARALLEL_OPENMP_FBXPLAT/g'",
            "-e 's/@AT_PARALLEL_NATIVE@/AT_PARALLEL_NATIVE_FBXPLAT/g'",
            "-e 's/@AT_PARALLEL_NATIVE_TBB@/AT_PARALLEL_NATIVE_TBB_FBXPLAT/g'",
            "-e 's/@AT_BUILD_WITH_LAPACK@/USE_LAPACK_FBXPLAT/g'",
            "-e 's/@AT_BLAS_F2C@/AT_BLAS_F2C_FBXPLAT/g'",
            "-e 's/@AT_BLAS_USE_CBLAS_DOT@/AT_BLAS_USE_CBLAS_DOT_FBXPLAT/g'",
            "aten/src/ATen/Config.h.in > $OUT/Config.h"
        ]),
        outs = {
            "Config.h": ["Config.h"],
        },
        default_outs = ["."],
    )

    fb_xplat_cxx_library(
        name = "generated_aten_headers_cpu",
        header_namespace = "ATen",
        exported_headers = get_aten_static_dispatch_backend_headers({
            "CPUFunctions.h": ":gen_aten[CPUFunctions.h]",
            "CPUFunctions_inl.h": ":gen_aten[CPUFunctions_inl.h]",
            "CompositeExplicitAutogradFunctions.h": ":gen_aten[CompositeExplicitAutogradFunctions.h]",
            "CompositeExplicitAutogradFunctions_inl.h": ":gen_aten[CompositeExplicitAutogradFunctions_inl.h]",
            "CompositeImplicitAutogradFunctions.h": ":gen_aten[CompositeImplicitAutogradFunctions.h]",
            "CompositeImplicitAutogradFunctions_inl.h": ":gen_aten[CompositeImplicitAutogradFunctions_inl.h]",
            "FunctionalInverses.h": ":gen_aten[FunctionalInverses.h]",
            "Functions.h": ":gen_aten[Functions.h]",
            "MethodOperators.h": ":gen_aten[MethodOperators.h]",
            "NativeFunctions.h": ":gen_aten[NativeFunctions.h]",
            "NativeMetaFunctions.h": ":gen_aten[NativeMetaFunctions.h]",
            "Operators.h": ":gen_aten[Operators.h]",
            "RedispatchFunctions.h": ":gen_aten[RedispatchFunctions.h]",
            "core/TensorBody.h": ":gen_aten[core/TensorBody.h]",
            "core/aten_interned_strings.h": ":gen_aten[core/aten_interned_strings.h]",
        }),
        feature = feature,
        labels = labels,
    )

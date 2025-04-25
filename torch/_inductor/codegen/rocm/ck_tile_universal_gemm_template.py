# mypy: allow-untyped-defs, disable-error-code="attr-defined, valid-type"
import logging
import random
from typing import List, Optional

import torch
from torch._inductor import config
from dataclasses import asdict, dataclass
from torch._inductor.codegen.rocm.ck_tile_template import CKTileTemplate
from torch._inductor.codegen.rocm.rocm_kernel import ROCmTemplateKernel
from torch._inductor.ir import Buffer, Layout

from ...utils import IndentedBuffer


log = logging.getLogger(__name__)


def is_static_int(number):
    import sympy
    return isinstance(number, (int, sympy.Integer))


def torch_layout_to_ck_layout(torch_layout):
    if torch_layout.stride[-1] == 1:
        return "Row"
    elif torch_layout.stride[-2] == 1:
        return "Col"
    else:
        return None

@dataclass
class CKTileGemmOperation:
    layout_a: str
    layout_b: str
    layout_c: str
    datatype: str
    
    tile_m: int
    tile_n: int
    tile_k: int

    warp_m: int
    warp_n: int
    warp_k: int

    warp_tile_m: int
    warp_tile_n: int
    warp_tile_k: int

    m_is_padded: bool
    n_is_padded: bool
    k_is_padded: bool

    pipeline: str
    scheduler: str
    epilogue: str

    def name(self):
        # TBD make unique and descriptive
        return "cktile_gemm_universal"


def _default_ops_list():
    return [
        CKTileGemmOperation(
            layout_a = "Row",
            layout_b = "Col",
            layout_c = "Row",
            datatype = "FP16",

            tile_m = 256,
            tile_n = 256,
            tile_k = 256,

            warp_m = 2,
            warp_n = 2,
            warp_k = 2,

            warp_tile_m = 32,
            warp_tile_n = 32,
            warp_tile_k = 16,

            m_is_padded = False,
            n_is_padded = False,
            k_is_padded = False,

            pipeline = "compv3",
            scheduler = "intrawave",
            epilogue = "default",
        )
    ]


class CKTileGemmTemplate(CKTileTemplate):
# the JINJA template for rendering CK Universal GEMMs
    gemm_template = r"""{{version_comment}}
    {{headers}}
    {{globals}}
    {{instance_definition}}
    extern "C" {
    PT_EXPORT {{kernel_definition}} {

        auto kargs = ck_tile::GemmKernelArgs {
           X,
           W,
           Y,
           M,
           N,
           K,
           LDA,
           LDB,
           LDC,
           kBatch
        };

        if (!{{instance_type}}::IsSupportedArgument(kargs)) {
            // we do our best to statically avoid this case in `filter_op`
            std::cerr << "invalid argument for gemm instance " << {{instance_type}}::GetName() 
                      << ", M: " << M << ", N: " << N << ", K: " << K 
                      << ", LDA: " << LDA << ", LDB: " << LDB << ", LDC: " << LDC 
                      << std::endl;
            return -45;
        }

        if (workspace_size) {
            *workspace_size = 0;
            return 0;
        }
        
        // run the kernel
        auto stream_config = ck_tile::stream_config{stream};
        auto grid_size = {{instance_type}}::GridSize(M, N, kBatch);
        auto block_size = {{instance_type}}::BlockSize();
        auto lds_bytes = 0;
        auto kBlockPerCU = 1;
        auto kernel = ck_tile::make_kernel<block_size.x, kBlockPerCU>({{instance_type}}{}, grid_size, block_size, lds_bytes, kargs);
        float elapsed_time = ck_tile::launch_kernel(stream_config, kernel);
        return 0;
    } // kernel definition
    } // extern C
    """

    def __init__(
        self,
        input_nodes: List[Buffer],
        layout: Layout,
    ) -> None:
        super().__init__(
            "ck_tile_gemm_template",
            input_nodes=input_nodes,
            layout=layout,
        )

    def header(self) -> IndentedBuffer:
        res = super().header()
        res.splice(
            """
                // CK GEMM header(s)

                #include "ck_tile/ops/gemm.hpp"
                #include "ck_tile/ops/epilogue.hpp"
            """
        )
        return res

    def globals(self) -> IndentedBuffer:
        res = super().globals()
        res.splice(
            """
                // CK GEMM globals

                using Row = ck_tile::tensor_layout::gemm::RowMajor;
                using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

            """
        )
        return res

    def filter_op(self, op: "CKTileGemmOperation"):
        """
        Determines whether a given op definition is suitable for the current
        input / output of the operation that this template implements.

        Filter is based on inputs' dtype, layout and statically inferred size.

        Returns None if the op is not suitable, otherwise returns the op to be used.
        """
        metas = [T.get_layout() for T in [*self.input_nodes, self.output_node]]
        X_meta = metas[0]
        W_meta = metas[1]
        Y_meta = metas[-1]
        # disable the instance if dtypes don't match
        if op.datatype != self._TORCH_DTYPE_TO_CK[X_meta.dtype]:
            return None
        if op.datatype != self._TORCH_DTYPE_TO_CK[W_meta.dtype]:
            return None
        if op.datatype != self._TORCH_DTYPE_TO_CK[Y_meta.dtype]:
            return None
        # disable the instance if layouts don't match
        if op.layout_a != torch_layout_to_ck_layout(X_meta):
            return None
        if op.layout_b != torch_layout_to_ck_layout(W_meta):
            return None
        if op.layout_c != torch_layout_to_ck_layout(Y_meta):
            return None
        return op
#         # try to avoid launching the instance with invalid problem size
#         # see GridwiseGemm_xdl_cshuffle_v3::CheckValidity

#         M = X_meta.size[-2]
#         K = X_meta.size[-1]
#         N = W_meta.size[-1]

#         if is_static_int(M):
#             if not any(
#                 m_padding in op.gemm_specialization
#                 for m_padding in ["MPadding", "MNPadding", "MKPadding", "MNKPadding"]
#             ):
#                 if M % op.m_per_block != 0:
#                     return None
#         if is_static_int(N):
#             if not any(
#                 n_padding in op.gemm_specialization
#                 for n_padding in ["NPadding", "MNPadding", "NKPadding", "MNKPadding"]
#             ):
#                 if N % op.n_per_block != 0:
#                     return None
#         if is_static_int(K):
#             if not any(
#                 k_padding in op.gemm_specialization
#                 for k_padding in ["KPadding", "MKPadding", "NKPadding", "MNKPadding"]
#             ):
#                 if K % op.k_per_block != 0:
#                     return None

#         a_contig_size = (
#             K if op.a_layout == "Row" else M if op.a_layout == "Col" else None
#         )
#         if (
#             is_static_int(a_contig_size)
#             and a_contig_size % op.a_block_transfer_src_scalar_per_vector != 0
#         ):
#             return None
#         b_contig_size = (
#             N if op.b_layout == "Row" else K if op.b_layout == "Col" else None
#         )
#         if (
#             is_static_int(b_contig_size)
#             and b_contig_size % op.b_block_transfer_src_scalar_per_vector != 0
#         ):
#             return None
#         c_contig_size = (
#             N if op.c_layout == "Row" else M if op.c_layout == "Col" else None
#         )
#         if (
#             is_static_int(c_contig_size)
#             and c_contig_size
#             % op.c_shuffle_block_transfer_scalar_per_vector_n_per_block
#             != 0
#         ):
#             return None

#         # TBD disable instances with invalid number of pipeline prefetch stages
#         # It will avoid compiling a small percentage of unrunnable instances which fail the gemm argument check

#         return op

    def emit_ck_instance(self, op: "CKTileGemmOperation"):
#         # The Jinja template for generating a C++ type alias *definition* for a Universal GEMM instance
        template_definition = r"""
    // Gemm operator {{operation_name}}

    namespace {{operation_name}} {

        constexpr int32_t TileM = {{tile_m}};
        constexpr int32_t TileN = {{tile_n}};
        constexpr int32_t TileK = {{tile_k}};

        constexpr int32_t WarpM = {{warp_m}};
        constexpr int32_t WarpN = {{warp_n}};
        constexpr int32_t WarpK = {{warp_k}};

        constexpr int32_t WarpTileM = {{warp_tile_m}};
        constexpr int32_t WarpTileN = {{warp_tile_n}};
        constexpr int32_t WarpTileK = {{warp_tile_k}};

        constexpr bool kPadM = {{m_is_padded}};
        constexpr bool kPadN = {{n_is_padded}};
        constexpr bool kPadK = {{k_is_padded}};

        using ALayout = {{layout_a}};
        using BLayout = {{layout_b}};
        using CLayout = {{layout_c}};

        using ADataType = {{datatype}};
        using BDataType = {{datatype}};
        using CDataType = {{datatype}};
        using AccDataType = F32;

        constexpr bool permuteA = false;
        constexpr bool permuteB = false;
        constexpr bool DoubleSmemBuffer = false;
        constexpr bool TransposeC = false;

        constexpr int kBlockPerCu                         = 1;
        constexpr ck_tile::index_t TileParitionerGroupNum = 8;
        constexpr ck_tile::index_t TileParitionerM01      = 4;

        using GemmShape = 
            ck_tile::TileGemmShape<ck_tile::sequence<TileM, TileN, TileK>,
                                   ck_tile::sequence<WarpM, WarpN, WarpK>,
                                   ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>,
                                   permuteA,
                                   permuteB>;

        using TilePartitioner =
            ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                      TileParitionerGroupNum,
                                                      TileParitionerM01>;

        using Traits  =
            ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout>;        

        using GemmUniversalTraits =
            ck_tile::TileGemmUniversalTraits<kPadM, kPadN, kPadK, DoubleSmemBuffer,
                                             ALayout, BLayout, CLayout, TransposeC>;    

        using GemmPipelineProblem =
            ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, GemmShape, Traits>;

        using BaseGemmPipeline = ck_tile::BaseGemmPipelineAgBgCr{{pipeline}}<GemmPipelineProblem>;  

        constexpr auto scheduler = ck_tile::GemmPipelineScheduler::{{scheduler}};

        constexpr bool has_hot_loop_v = true;
        constexpr int32_t tail_number_v = 1;

        using UniversalGemmProblem = 
            ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                  BDataType,
                                                  AccDataType,
                                                  GemmShape,
                                                  GemmUniversalTraits,
                                                  scheduler,
                                                  has_hot_loop_v,
                                                  tail_number_v>;

        using GemmPipeline = ck_tile::GemmPipelineAgBgCr{{pipeline}}<UniversalGemmProblem>;  

        using GemmEpilogue = ck_tile::DefaultGemm2DEpilogue<
                                ck_tile::DefaultGemm2DEpilogueProblem<AccDataType, 
                                                                      CDataType, 
                                                                      CLayout, 
                                                                      kPadM,
                                                                      kPadN,
                                                                      WarpTileM,
                                                                      WarpTileN,
                                                                      WarpTileK,
                                                                      UniversalGemmProblem::TransposeC>>;

        using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
    }

"""
        # The Jinja template for generating a C++ type alias *usage* for a Universal GEMM instance
        template_type = r"""{{operation_name}}::Kernel"""

        rendered_definition = self._template_from_string(template_definition).render(
            operation_name=op.name(),
            **asdict(op).items()
        )
        rendered_type = self._template_from_string(template_type).render(
            operation_name=op.name()
        )
        return (rendered_definition, rendered_type)

    def render(self, kernel: ROCmTemplateKernel, op: "CKTileGemmOperation", **kwargs) -> str:  # type: ignore[override]
        """
        The primary entry point for the code rendering process used in this template.
        """
        epilogue_nodes = kwargs.get("epilogue_nodes", None)
        assert epilogue_nodes is None or 0 == len(epilogue_nodes)
        template_buffer_node = kwargs.get("template_buffer_node", None)
        if template_buffer_node is not None:
            self.output_node = template_buffer_node
        assert 2 == len(self.input_nodes)
        X, W = self.input_nodes
        Y = self.output_node

        instance_definition, instance_type = self.emit_ck_instance(op)

        version_comment = rf"""/**
* Generated code for CK inductor backend
* See {type(self).__module__}.{type(self).__qualname__}
*
* Template instance {op}
*
* {torch.__version__=}
* torch.version.git_version={getattr(torch.version, 'git_version', 'None')}
*/
"""

        return self._template_from_string(self.gemm_template).render(
            headers=self.header().getvalue(),
            globals=self.globals().getvalue(),
            instance_definition=instance_definition,
            kernel_definition=kernel.def_kernel(
                inputs=[X, W],  # type: ignore[list-item]
                outputs=[Y],
                names_str="X, W, Y",
                size_args=[
                    f"int32_t {arg}"
                    for arg in ["M", "N", "K", "LDA", "LDB", "LDC"]
                ],
            ),
            instance_type=instance_type,
            version_comment=version_comment,
        )

#     def _is_rcr_f16(self):
#         X_meta, W_meta, Y_meta = (
#             T.get_layout() for T in [*self.input_nodes, self.output_node]
#         )
#         X_dtype, W_dtype, Y_dtype = (
#             self._TORCH_DTYPE_TO_CK[m.dtype] for m in (X_meta, W_meta, Y_meta)
#         )
#         X_layout, W_layout, Y_layout = (
#             torch_layout_to_ck_layout(m) for m in (X_meta, W_meta, Y_meta)
#         )

#         return (
#             X_dtype == "F16"
#             and W_dtype == "F16"
#             and Y_dtype == "F16"
#             and X_layout == "Row"
#             and W_layout == "Col"
#             and Y_layout == "Row"
#         )

    def gen_ops(self):
        """
        Creates a list of `CKTileGemmOperation` instances that match the GEMM operation this template represents.
        The instances are guaranteed to have the correct layout, dtype and dimension padding for the GEMM input arguments.

        An instance may invalidate the GEMM configuration at runtime.
        Such instances will be assigned +inf runtime by the autotune process.
        """
        unfiltered_instances = _default_ops_list()
        filtered_instances = list(
            filter(lambda op: self.filter_op(op), unfiltered_instances)
        )
        # NB: when using a fixed list order, most likely we will pick the subset of instances
        # which are very similar to each other. Randomizing the choice seems to solve this.
        random.seed(-11)
        chosen_instances = (
            random.sample(
                filtered_instances,
                min(len(filtered_instances), config.rocm.n_max_profiling_configs),
            )
            if config.rocm.n_max_profiling_configs
            else filtered_instances
        )
        log.debug(
            "generated %d ck instances after filter: %s",
            len(chosen_instances),
            chosen_instances,
        )
        return chosen_instances

    @staticmethod
    def add_choices(
        choices,
        layout,
        input_nodes,
    ):
        """
        Add Composable Kernel Universal GEMM instance choices to the auto-tuning list.
        """
        template = CKTileGemmTemplate(
            input_nodes,
            layout,
        )
        ops = template.gen_ops()
        for op in ops:
            template.maybe_append_choice(
                choices,
                op=op,
            )

    def size_args(self):
        X = self.input_nodes[0]
        W = self.input_nodes[1]
        Y = self.output_node

        M = X.get_size()[0]
        K = X.get_size()[1]
        N = W.get_size()[1]
        LDA = X.get_stride()[0 if X.get_stride()[1] == 1 else 1]
        LDB = W.get_stride()[0 if W.get_stride()[1] == 1 else 1]
        LDC = Y.get_stride()[0 if Y.get_stride()[1] == 1 else 1]

        return M, N, K, LDA, LDB, LDC

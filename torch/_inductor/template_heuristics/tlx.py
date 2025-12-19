from torch._inductor import config

if config.is_fbcode() and config.triton.enable_tlx_templates:
        import torch._inductor.fb.tlx_templates.registry # noqa: F401

# TODO. Move the registry to this file once the TLX template is more complete.

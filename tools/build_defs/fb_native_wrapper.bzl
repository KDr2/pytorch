# Only used for PyTorch open source BUCK build

def _genrule(default_outs = ["."], **kwargs):
    genrule(
        # default_outs is only needed for internal BUCK
        **kwargs
    )

def _read_config(**kwargs):
    read_config(**kwargs)

fb_native = struct(
    genrule = _genrule,
    read_config = _read_config,
)

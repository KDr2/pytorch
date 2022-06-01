# Only used for PyTorch open source BUCK build

def fb_xplat_genrule(default_outs = ["."], **kwargs):
    genrule(
        # default_outs=default_outs, # only needed for internal BUCK
        **kwargs
    )

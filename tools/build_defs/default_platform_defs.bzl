# Only used for PyTorch open source BUCK build

def compose_platform_setting_list(settings):
    """Settings object:
    os/cpu pair: should be valid key, or at most one part can be wildcard.
    flags: the values added to the compiler flags
    """
    result = []
    for setting in settings:
        result = result.append([
            "^{}-{}$".format(setting["os"], setting["cpu"]),
            setting["flags"],
        ])
    return result

# Only used for PyTorch open source BUCK build

IGNORED_ATTRIBUTE_PREFIX = [
    "apple",
    "fbobjc",
    "windows",
    "fbandroid",
    "macosx",
]

IGNORED_ATTRIBUTES = [
    "feature",
    "platforms",
]

def filter_attributes(kwgs):
    keys = list(kwgs.keys())

    # drop unncessary attributes
    for key in keys:
        if key in IGNORED_ATTRIBUTES:
            kwgs.pop(key)
        else:
            for invalid_prefix in IGNORED_ATTRIBUTE_PREFIX:
                if key.startswith(invalid_prefix):
                    kwgs.pop(key)
    return kwgs

# map fbsource deps to OSS deps
def to_oss_deps(deps = []):
    new_deps = []
    for dep in deps:
        new_deps = new_deps + process_deps(dep)
    return new_deps

def process_deps(dep):
    # remove @fbsource prefix
    if dep.startswith("@fbsource"):
        dep = dep[len("@fbsource"):]

    # remove xplat/caffe2 prefffix
    if dep.startswith("//xplat/caffe2"):
        dep = dep[len("//xplat/caffe2"):]

    if dep.startswith("//third-party/"):
        dep = dep[len("//third-party/"):]
        target_name = dep.split(":")[1]
        return ["//third_party:" + target_name]

    if dep.startswith("//xplat/third-party/"):
        dep = dep[len("//xplat/third-party/"):]
        target_name = dep.split(":")[1]
        return ["//third_party:" + target_name]

    return [dep]

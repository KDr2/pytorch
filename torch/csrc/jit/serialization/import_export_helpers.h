#pragma once

#include <memory>
#include <string>

namespace torch {
namespace serialize {
class PyTorchStreamReader;
}
} // namespace torch

namespace torch {
namespace jit {

struct Source;

// Convert a class type's qualifier name to the corresponding path the source
// file it should be written to.
//
// Qualifier is like: foo.bar.baz
// Returns: libs/foo/bar/baz.py
std::string qualifierToArchivePath(
    const std::string& qualifier,
    const std::string& export_prefix);

std::shared_ptr<Source> findSourceInArchiveFromQualifier(
    torch::serialize::PyTorchStreamReader& reader,
    const std::string& export_prefix,
    const std::string& qualifier);

} // namespace jit
} // namespace torch

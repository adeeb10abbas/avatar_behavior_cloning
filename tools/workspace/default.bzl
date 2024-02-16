load("//tools/workspace/bazel_deps:repository.bzl", "bazel_deps_repository")
load("//tools/workspace/pytorch:repository.bzl", "pytorch_repository")
load("//tools/workspace/diffusion_policy:repository.bzl", "diffusion_repository")
load("//tools/workspace/footpedal:repository.bzl", "footpedal_repository")

def add_default_repositories(excludes = []):
    if "bazel_deps" not in excludes:
        bazel_deps_repository()

def add_essentials():
    pytorch_repository(name = "pytorch")
    diffusion_repository(name = "diffusion_policy_py")
    footpedal_repository(name = "foot")


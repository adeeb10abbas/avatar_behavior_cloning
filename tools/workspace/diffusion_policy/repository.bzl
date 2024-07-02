load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def add_diffusion_repository(name):
    http_archive(
        name = "diffusion_policy",
        urls = ["https://github.com/real-stanford/diffusion_policy/archive/548a52bbb105518058e27bf34dcf90bf6f73681a.tar.gz"],
        sha256 = "a4c9a488ed9565357fbab38b866608247dfc35fd34eb37ec12552b3a5594d072",
        build_file = "//tools/workspace/diffusion_policy:package.BUILD.bazel",
        strip_prefix = "diffusion_policy-548a52bbb105518058e27bf34dcf90bf6f73681a/diffusion_policy",
    )
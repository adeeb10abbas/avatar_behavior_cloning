load("@drake//tools/workspace:github.bzl", "github_archive")

def diffusion_repository(name, mirrors = None):
    github_archive(
        name = name,
        repository = "real-stanford/diffusion_policy",
        commit = "548a52bbb105518058e27bf34dcf90bf6f73681a",
        sha256 = "a4c9a488ed9565357fbab38b866608247dfc35fd34eb37ec12552b3a5594d072",
        upgrade_advice = "",
        build_file = "//tools/workspace/diffusion:package.BUILD.bazel",
        mirrors = {"github": ["https://github.com/{repository}/archive/{commit}.tar.gz"]},
    )
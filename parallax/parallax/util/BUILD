licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//visibility:public",
    ],
)

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = [
        "//parallax:parallax",
        "//parallax/core/python/tools:tools",
    ],
)

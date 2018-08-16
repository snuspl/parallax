licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
         "//visibility:public",
    ],
)

native.py_library(
    name = "python",
    srcs = ["__init__.py"],
    deps = [
        "//parallax/core/python/common:common",
        "//parallax/core/python/mpi:mpi",
        "//parallax/core/python/ps:ps",
        "//parallax/core/python/hybrid:hybrid",
        "//parallax/core/python/tools:tools",
    ],
)

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
         "//visibility:public",
    ],
)

native.py_library(
    name = "parallax",
    srcs = ["__init__.py"],
    deps = [
        "//parallax/core:core",
        "//parallax/core/python/common:runner",
        "//parallax/core/python/common:shard",
    ],
)
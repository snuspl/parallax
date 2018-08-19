licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
         "//visibility:public",
    ],
)

native.py_library(
    name = "graph_transform",
    srcs = ["graph_transform.py"],
    deps = [
        "//parallax/core/python/common:graph_transform_lib",
        "//parallax/core/python/common:lib",
    ]
)

native.py_library(
    name = "runner",
    srcs = ["runner.py"],
    deps = [
        "graph_transform",
        "//parallax/core/python/common:lib",
        "//parallax/core/python/common:consts",
        "//parallax/core/python/common:session_context",
    ]
)

native.py_library(
    name = "mpi",
    srcs = ["__init__.py"],
    deps = [
        "runner"
    ]
)

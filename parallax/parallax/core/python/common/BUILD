licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
         "//visibility:public",
    ],
)



native.py_library(
    name = "lib",
    srcs = ["lib.py"],
    deps = [
        "consts",
    ]
)

native.py_library(
    name = "config",
    srcs = ["config.py"],
    deps = [
    ]
)

native.py_library(
    name = "graph_transform_lib",
    srcs = ["graph_transform_lib.py"],
    deps = [
        "lib",
    ]
)

native.py_library(
    name = "session_context",
    srcs = ["session_context.py"],
    deps = [
    ]
)

native.py_library(
    name = "runner",
    srcs = ["runner.py"],
    deps = [
        "lib",
        "graph_transform_lib",
        "consts",
        "//parallax/core/python/ps:runner",
        "//parallax/core/python/mpi:runner",
        "//parallax/core/python/hybrid:runner"
    ]
)

native.py_library(
    name = "shard",
    srcs = ["shard.py"],
    deps = [
        "graph_transform_lib",
    ],
)

native.py_library(
    name = "consts",
    srcs = ["consts.py"],
    deps = [
    ],
)

native.py_library(
    name = "common",
    srcs = ["__init__.py"],
    deps = [
        "graph_transform_lib",
        "runner",
        "shard",
        "config",
        "session_context",
    ],
)


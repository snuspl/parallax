licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
         "//visibility:public",
    ],
)

native.py_library(
    name = "launch_ps",
    srcs = ["launch_ps.py"]
)

native.py_library(
    name = "tools",
    srcs = ["__init__.py"],
    deps = [
        "launch_ps"
    ],
)

rm ~/wheel/*parallax*.whl
pip uninstall parallax -y

bazel build --config=opt --config=cuda //parallax/util:build_pip_package
bazel-bin/parallax/util/build_pip_package ~/wheel
pip install ~/wheel/parallax*.whl

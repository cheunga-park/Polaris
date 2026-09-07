"""Run a 2DGS repository script with the CUDA kernels we built for the evaluation renderer.

2DGS's ``scene/gaussian_model.py`` does ``from simple_knn._C import distCUDA2`` — it expects
``_C`` to be a *submodule*. Our (upstream-polaris) ``simple_knn`` JIT-loads the extension and
keeps it as a package attribute. Registering that object under the submodule name is the
whole bridge; the 2DGS checkout is not edited.

    python -m polaris_v2s.run_2dgs train.py -s <dataset> -m <model> ...   (cwd = 2DGS repo)
"""

from __future__ import annotations

import runpy
import sys


def main() -> None:
    import simple_knn
    sys.modules.setdefault("simple_knn._C", simple_knn._C)
    import diff_surfel_rasterization  # noqa: F401  (JIT-compiles on first import if needed)
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()

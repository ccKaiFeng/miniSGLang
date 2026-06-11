# 这个文件是 `python -m minisgl.kernel` 的入口。
#
# 它用于生成 `.clangd`，让 clangd 能理解本项目的 CUDA/C++ 扩展 include 路径
# 和 GPU 架构参数，方便编辑器做跳转、补全和语法检查。

assert __name__ == "__main__"


def generate_clangd():
    """根据当前 GPU compute capability 生成 .clangd 配置。"""

    import os
    import subprocess

    from minisgl.kernel.utils import DEFAULT_INCLUDE
    from minisgl.utils import init_logger
    from tvm_ffi.libinfo import find_dlpack_include_path, find_include_path

    logger = init_logger(__name__)
    logger.info("Generating .clangd file...")

    # TVM FFI 和 DLPack 的 include 路径需要加入 clangd，否则头文件无法解析。
    include_paths = [find_include_path(), find_dlpack_include_path()] + DEFAULT_INCLUDE

    # 读取当前 GPU 的 compute capability，例如 9.0 -> sm_90。
    status = subprocess.run(
        args=["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True,
        check=True,
    )
    compute_cap = status.stdout.decode("utf-8").strip().split("\n")[0]
    major, minor = compute_cap.split(".")
    compile_flags = ",\n    ".join(
        [
            "-xcuda",
            f"--cuda-gpu-arch=sm_{major}{minor}",
            "-std=c++20",
            "-Wall",
            "-Wextra",
        ]
        + [f"-isystem{path}" for path in include_paths]
    )
    clangd_content = f"""
CompileFlags:
  Add: [
    {compile_flags}
  ]
"""
    if os.path.exists(".clangd"):
        # 已存在时不覆盖，避免改掉用户自己的 clangd 配置。
        logger.warning(".clangd file already exists, nothing done.")
        logger.warning(f"suggested content: {clangd_content}")
    else:
        # 只在没有 .clangd 时新建。
        with open(".clangd", "w") as f:
            f.write(clangd_content)
        logger.info(".clangd file generated.")


generate_clangd()

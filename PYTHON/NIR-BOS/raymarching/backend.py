import os
from torch.utils.cpp_extension import load

_src_path = os.path.dirname(os.path.abspath(__file__))

nvcc_flags = [
    '-O3', '-std=c++17',
    '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
]

if os.name == "posix":
    c_flags = ['-O3', '-std=c++17']
elif os.name == "nt":
    c_flags = ['/O2', '/std:c++17']

    # find cl.exe
    def find_cl_path():
        import glob, subprocess
        # Strategy 1: use vswhere.exe (locates VS 2017+ regardless of install drive)
        vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
        if os.path.exists(vswhere):
            try:
                vs_path = subprocess.check_output(
                    [vswhere, "-latest", "-products", "*",
                     "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                     "-property", "installationPath"], text=True).strip()
                if vs_path:
                    msvc_glob = os.path.join(vs_path, "VC", "Tools", "MSVC", "*", "bin", "Hostx64", "x64")
                    paths = sorted(glob.glob(msvc_glob), reverse=True)
                    if paths:
                        return paths[0]
            except Exception:
                pass
        # Strategy 2: scan C: and D: drives
        for root in [r"C:\\", r"D:\\"]:
            for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
                pattern = os.path.join(root, "Program Files*", "Microsoft Visual Studio", "*", edition,
                                       "VC", "Tools", "MSVC", "*", "bin", "Hostx64", "x64")
                paths = sorted(glob.glob(pattern), reverse=True)
                if paths:
                    return paths[0]
        return None

    # If cl.exe is not on path, try to find it.
    if os.system("where cl.exe >nul 2>nul") != 0:
        cl_path = find_cl_path()
        if cl_path is None:
            raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
        os.environ["PATH"] += ";" + cl_path

_backend = load(name='_raymarching',
                extra_cflags=c_flags,
                extra_cuda_cflags=nvcc_flags,
                sources=[os.path.join(_src_path, 'src', f) for f in [
                    'raymarching.cu',
                    'bindings.cpp',
                ]],
                verbose=True
                )

__all__ = ['_backend']
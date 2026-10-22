# Build a CPU-only CTranslate2 Python wheel for Windows x64 from the exact upstream tag.
#
# The PyPI wheel statically links Intel MKL, ships Intel's libiomp5md.dll and an unused
# cudnn64_9.dll (python/tools/prepare_build_environment_windows.sh upstream). This build
# uses only open-source backends: oneDNN (Apache-2.0, static) and the MSVC OpenMP runtime.
#
# Output directory ($Out):
#   ctranslate2-<ver>-*.whl                 wheel installed into the BookTrans build
#   ctranslate2-<ver>-recursive-src.tar.gz  tag + every submodule, no .git (corresponding source)
#   oneDNN-<ver>.tar.gz                     upstream oneDNN source used for the static library
#   SUBMODULES.txt, BUILD_INFO.txt          exact commits and tool versions
param(
    [string]$Version = "4.8.3",
    [string]$OneDnnVersion = "3.1.1",
    [string]$Out = "ct2-dist"
)
$ErrorActionPreference = "Stop"
function Check([string]$what) { if ($LASTEXITCODE -ne 0) { throw "$what failed ($LASTEXITCODE)" } }

$Out = New-Item -ItemType Directory -Force $Out | Select-Object -ExpandProperty FullName
$work = New-Item -ItemType Directory -Force "ct2-work" | Select-Object -ExpandProperty FullName
Push-Location $work
try {
    git -c advice.detachedHead=false clone --branch "v$Version" --depth 1 https://github.com/OpenNMT/CTranslate2 src; Check "clone"
    git -C src submodule update --init --recursive; Check "submodules"
    $commit = git -C src rev-parse HEAD
    "CTranslate2 v$Version $commit" | Set-Content "$Out/SUBMODULES.txt" -Encoding utf8
    git -C src submodule status --recursive | Add-Content "$Out/SUBMODULES.txt" -Encoding utf8
    python "$PSScriptRoot/archive_tree.py" src "$Out/ctranslate2-$Version-recursive-src.tar.gz" "ctranslate2-$Version"; Check "source archive"

    $dnnlTar = "$Out/oneDNN-$OneDnnVersion.tar.gz"
    curl.exe -fsSL -o $dnnlTar "https://github.com/oneapi-src/oneDNN/archive/refs/tags/v$OneDnnVersion.tar.gz"; Check "oneDNN download"
    tar -xf $dnnlTar; Check "oneDNN extract"
    # Same oneDNN options as the upstream Windows wheel build.
    cmake -S "oneDNN-$OneDnnVersion" -B dnnl-build -DCMAKE_INSTALL_PREFIX="$work/dnnl-install" `
        -DONEDNN_LIBRARY_TYPE=STATIC -DONEDNN_BUILD_EXAMPLES=OFF -DONEDNN_BUILD_TESTS=OFF `
        -DONEDNN_ENABLE_WORKLOAD=INFERENCE -DONEDNN_ENABLE_PRIMITIVE="CONVOLUTION;REORDER" -DONEDNN_BUILD_GRAPH=OFF; Check "oneDNN configure"
    cmake --build dnnl-build --config Release --target install --parallel; Check "oneDNN build"

    cmake -S src -B ct2-build -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release `
        -DCMAKE_INSTALL_PREFIX="$work/ct2-install" -DCMAKE_PREFIX_PATH="$work/dnnl-install" `
        -DBUILD_CLI=OFF -DWITH_MKL=OFF -DWITH_DNNL=ON -DWITH_CUDA=OFF -DWITH_CUDNN=OFF -DOPENMP_RUNTIME=COMP; Check "CTranslate2 configure"
    cmake --build ct2-build --config Release --target install --parallel; Check "CTranslate2 build"

    Copy-Item "$work/ct2-install/bin/ctranslate2.dll" src/python/ctranslate2/
    python -m pip install pybind11==2.13.6 setuptools wheel; Check "wheel build deps"
    $env:CTRANSLATE2_ROOT = "$work/ct2-install"
    python -m pip wheel src/python --no-deps --no-build-isolation -w $Out; Check "wheel"

    @(
        "ctranslate2 $Version commit $commit",
        "options: WITH_MKL=OFF WITH_DNNL=ON WITH_CUDA=OFF WITH_CUDNN=OFF OPENMP_RUNTIME=COMP BUILD_CLI=OFF",
        "oneDNN $OneDnnVersion static (INFERENCE; CONVOLUTION;REORDER)",
        (cmake --version | Select-Object -First 1),
        "python $(python -c 'import sys; print(sys.version)')",
        (python -m pip freeze | Select-String -Pattern '^(pybind11|setuptools|wheel)==')
    ) | Out-String | Set-Content "$Out/BUILD_INFO.txt" -Encoding utf8
} finally {
    Pop-Location
}
Get-ChildItem $Out | Format-Table Name, Length

//! Build script for the `aimd-mlx` C-ABI shim (WS-11a).
//!
//! Discovers the brew `mlx` install (formula `mlx`, 0.32.4 bottled; override
//! with `MLX_PREFIX`), compiles `shim/shim.cpp` against its headers with the
//! `cc` crate, and links `libmlx.dylib`. An rpath to the mlx lib dir is
//! embedded so test/link products also resolve `libmlx.dylib` and its
//! `@rpath/libjaccl.dylib` dependency without `DYLD_*` env fiddling; the
//! bundled `mlx.metallib` is located by MLX relative to the loaded dylib.
//!
//! On Linux (spec 003) the prefix is an mlx build with the CUDA backend
//! (`MLX_PREFIX`, e.g. the `mlx/` dir of the `mlx-cuda-13` wheel): it links
//! `libmlx.so` against libstdc++ with the same rpath.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=MLX_PREFIX");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=shim/shim.cpp");

    let prefix = mlx_prefix();
    let include_dir = prefix.join("include");
    let lib_dir = prefix.join("lib");

    let linux = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux");
    let header = include_dir.join("mlx").join("ops.h");
    let lib = lib_dir.join(if linux { "libmlx.so" } else { "libmlx.dylib" });
    for required in [&header, &lib] {
        if !required.is_file() {
            panic!(
                "mlx install at {} is incomplete: missing {} \
                 (install with `brew install mlx` or set MLX_PREFIX)",
                prefix.display(),
                required.display()
            );
        }
    }
    // Rebuild when the linked dylib changes (brew upgrades).
    println!("cargo:rerun-if-changed={}", lib.display());

    cc::Build::new()
        .cpp(true)
        // mlx 0.32.4 headers use C++20 (e.g. defaulted comparison operators
        // in mlx/backend/common/metal_kernel.h).
        .std("c++20")
        .cpp_link_stdlib(if linux { "stdc++" } else { "c++" })
        .file("shim/shim.cpp")
        .include(&include_dir)
        .compile("aimd_mlx_shim");

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=mlx");
    // Resolve libmlx itself and its `@rpath/libjaccl.dylib` dependency at
    // runtime without environment setup.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
}

/// MLX install prefix: `MLX_PREFIX` wins, else `brew --prefix mlx`.
fn mlx_prefix() -> PathBuf {
    if let Ok(prefix) = std::env::var("MLX_PREFIX") {
        if prefix.is_empty() {
            panic!("MLX_PREFIX is set but empty");
        }
        return PathBuf::from(prefix);
    }
    let output = Command::new("brew")
        .args(["--prefix", "mlx"])
        .output()
        .unwrap_or_else(|e| {
            panic!(
                "cannot run `brew --prefix mlx` ({e}); install mlx with \
                 `brew install mlx` or set MLX_PREFIX to the mlx prefix"
            )
        });
    if !output.status.success() {
        panic!(
            "`brew --prefix mlx` failed ({}); install mlx with `brew install \
             mlx` or set MLX_PREFIX",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let prefix = String::from_utf8(output.stdout).expect("brew output is utf-8");
    let prefix = PathBuf::from(prefix.trim());
    if !prefix.is_dir() {
        panic!(
            "`brew --prefix mlx` returned {}, which is not a directory; \
             install mlx with `brew install mlx` or set MLX_PREFIX",
            prefix.display()
        );
    }
    prefix
}

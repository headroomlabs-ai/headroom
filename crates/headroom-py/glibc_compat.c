/*
 * glibc < 2.38 compatibility shim for the C23 strtol* family.
 *
 * Why this file exists
 * --------------------
 *
 * glibc 2.38 (Aug 2023) added `__isoc23_strtol`, `__isoc23_strtoll`,
 * `__isoc23_strtoul`, and `__isoc23_strtoull` as canonical C23
 * implementations of strtol*. When you compile C/C++ code with a
 * recent toolchain (gcc >= 13) and the headers see C23/C++23 mode
 * (or `_GNU_SOURCE`), `<stdlib.h>` redirects every call to
 * `strtoll(...)` to `__isoc23_strtoll(...)` via a transparent
 * `__REDIRECT_NTH` attribute.
 *
 * The ONNX Runtime prebuilt artifacts that we statically link
 * (downloaded by `ort-download-binaries-rustls-tls` via fastembed)
 * are compiled with gcc-14.2.1 on a glibc-2.38+ host. They
 * therefore reference `__isoc23_*` symbols. Our wheel build runs
 * in `manylinux_2_28` (glibc 2.28 baseline), so the link is fine
 * — the linker doesn't validate that ALL referenced symbols
 * exist in the target glibc, only that the SONAME matches.
 *
 * On the END USER's runtime, however, glibc < 2.38 has none of
 * these symbols, and `import headroom._core` fails with:
 *
 *     ImportError: undefined symbol: __isoc23_strtoll
 *
 * (Reported in issue #355; first hit by users on Ubuntu 22.04 +
 * Conda Python 3.12 environments where libc.so.6 is glibc 2.35.)
 *
 * The fix
 * -------
 *
 * Provide local, statically-linked-into-_core.so definitions of the
 * four `__isoc23_*` symbols that delegate to the older `strtol*`
 * family (which exists in EVERY glibc the manylinux_2_28 floor
 * targets). The static linker resolves ORT's UND `__isoc23_*`
 * references against these definitions inside `_core.so`.
 *
 * Two implementation traps to avoid (both bit PR #384's first iter):
 *
 * 1. NO `__attribute__((alias("strtol")))`. GCC requires the alias
 *    target to be defined in the same translation unit; `strtol` is
 *    in libc.so.6, NOT this .c file, so GCC errors at compile time:
 *      glibc_compat.c: error: '__isoc23_strtol' aliased to undefined symbol 'strtol'
 *
 * 2. NO `#include <stdlib.h>`. On the manylinux_2_28 build host the
 *    toolchain is recent enough that `<stdlib.h>` may apply the
 *    `__REDIRECT_NTH(strtol, ..., __isoc23_strtol)` rewrite when
 *    `_GNU_SOURCE` is implicit. If we included it, our call to
 *    `strtol(...)` inside `__isoc23_strtol` would be silently
 *    rewritten to call `__isoc23_strtol` itself — infinite recursion
 *    on glibc 2.38+ and stack overflow on first use. Forward-declare
 *    the older POSIX prototypes ourselves so the call goes to the
 *    actual unredirected symbol.
 *
 * Behavioural caveat
 * ------------------
 *
 * The C23 `__isoc23_strtoll` accepts binary-literal input ("0b1010")
 * which the older `strtoll` rejects. Our call sites are deep inside
 * ORT's protobuf parsing, which only feeds decimal/hex strings, so
 * the fallback is functionally identical for our use. If a future
 * statically-linked library DOES depend on binary-literal parsing
 * we'd need to reimplement the parser; for now this shim is sound.
 *
 * Symbol-resolution semantics
 * ---------------------------
 *
 * Once `_core.so` is dlopen'd by Python, lookups of `__isoc23_strtoll`
 * by code inside `_core.so` go through the dynamic linker. On glibc
 * 2.38+, libc.so.6 (which is in the global scope, loaded by
 * Python's executable) has the strong symbol — it wins, our local
 * definition is shadowed but harmless. On glibc < 2.38, libc.so.6
 * has no such symbol; the dynamic linker falls back to
 * `_core.so`'s local symbol — ours wins. Either way, the symbol
 * resolves and `import headroom._core` succeeds.
 *
 * References:
 * - https://sourceware.org/glibc/wiki/Release/2.38
 * - https://github.com/pypa/manylinux/issues/1725
 * - issue #355: tests/test_rust_core_smoke.py was the canary that
 *   surfaced this on user installs.
 *
 * This file is compiled and linked into `_core.so` only on Linux
 * with the GNU libc env (gated in build.rs). macOS and Windows
 * have neither glibc nor this symbol family.
 */

/*
 * Forward declarations for the actual (pre-C23) glibc strtol family.
 * Deliberately NOT pulled from <stdlib.h> — see trap #2 above.
 * These prototypes match POSIX 1003.1-2008.
 */
extern long strtol(const char *nptr, char **endptr, int base);
extern long long strtoll(const char *nptr, char **endptr, int base);
extern unsigned long strtoul(const char *nptr, char **endptr, int base);
extern unsigned long long strtoull(const char *nptr, char **endptr, int base);

long __isoc23_strtol(const char *nptr, char **endptr, int base) {
    return strtol(nptr, endptr, base);
}

long long __isoc23_strtoll(const char *nptr, char **endptr, int base) {
    return strtoll(nptr, endptr, base);
}

unsigned long __isoc23_strtoul(const char *nptr, char **endptr, int base) {
    return strtoul(nptr, endptr, base);
}

unsigned long long __isoc23_strtoull(const char *nptr, char **endptr, int base) {
    return strtoull(nptr, endptr, base);
}

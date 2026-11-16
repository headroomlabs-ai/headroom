"""Windows NTFS ACL implementation backing :mod:`headroom.fileperms`.

Python ships no ACL API, so this module talks to ``advapi32``/``kernel32``
through ``ctypes`` directly rather than adding a third-party dependency
(``pywin32``) for a handful of calls. Imported only on Windows; every public
function fails closed (returns ``False``) on any error rather than raising,
because the caller's contract (see :mod:`headroom.fileperms`) is "report
whether owner-only protection took effect" — a partially-applied ACL that
raised out of a logging setup path would be worse than one that silently
reports failure and lets the caller warn.

Both entry points restrict a HANDLE, not a path: :func:`restrict_handle` for
a file we hold open (its fd's handle is reopened via ``ReOpenFile`` for
``WRITE_DAC``, which the CRT's own open does not request, with no path
re-resolution and therefore no TOCTOU window), and :func:`restrict_path` for
one we did not (rotated backups) — opened with ``FILE_FLAG_OPEN_REPARSE_POINT``
so a symlink/junction at the path is restricted as the link itself rather
than followed, mirroring the POSIX ``O_NOFOLLOW`` used for the same purpose.

The DACL this sets has exactly one entry: full control for the calling
user's own SID, created fresh (not merged with the inherited DACL) and
applied with ``PROTECTED_DACL_SECURITY_INFORMATION`` so the parent
directory's inherited entries — which is what made the file readable in the
first place — are dropped rather than left alongside the new entry. That is
deliberately stricter than POSIX ``0600`` relative to an administrator:
POSIX root bypasses file permissions outright, while this ACL does not carve
out Administrators/SYSTEM, matching what "owner-only" says.
"""

from __future__ import annotations

import ctypes
import msvcrt
import os
from ctypes import wintypes

_advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)  # type: ignore[attr-defined]
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]

_SE_FILE_OBJECT = 1
_DACL_SECURITY_INFORMATION = 0x00000004
_PROTECTED_DACL_SECURITY_INFORMATION = 0x80000000
_NO_INHERITANCE = 0
_SET_ACCESS = 2
_ERROR_SUCCESS = 0
_FILE_ALL_ACCESS = 0x001F01FF
_TOKEN_QUERY = 0x0008
_TokenUser = 1
_WRITE_DAC = 0x00040000
_READ_CONTROL = 0x00020000
_OPEN_EXISTING = 3
_CREATE_NEW = 1
_GENERIC_WRITE = 0x40000000
_ERROR_FILE_NOT_FOUND = 2
_FILE_BEGIN = 0
_FILE_ATTRIBUTE_NORMAL = 0x80
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_FILE_SHARE_READ = 1
_FILE_SHARE_WRITE = 2
_FILE_SHARE_DELETE = 4
_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value


class _TRUSTEE_W(ctypes.Structure):
    pass


_TRUSTEE_W._fields_ = [
    ("pMultipleTrustee", ctypes.POINTER(_TRUSTEE_W)),
    ("MultipleTrusteeOperation", ctypes.c_int),
    ("TrusteeForm", ctypes.c_int),
    ("TrusteeType", ctypes.c_int),
    ("ptstrName", ctypes.c_void_p),
]


class _EXPLICIT_ACCESS_W(ctypes.Structure):
    _fields_ = [
        ("grfAccessPermissions", wintypes.DWORD),
        ("grfAccessMode", ctypes.c_int),
        ("grfInheritance", wintypes.DWORD),
        ("Trustee", _TRUSTEE_W),
    ]


_advapi32.BuildTrusteeWithSidW.argtypes = [ctypes.POINTER(_TRUSTEE_W), ctypes.c_void_p]
_advapi32.BuildTrusteeWithSidW.restype = None

_advapi32.SetEntriesInAclW.argtypes = [
    wintypes.ULONG,
    ctypes.POINTER(_EXPLICIT_ACCESS_W),
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
]
_advapi32.SetEntriesInAclW.restype = wintypes.DWORD

_advapi32.SetSecurityInfo.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    wintypes.DWORD,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]
_advapi32.SetSecurityInfo.restype = wintypes.DWORD

_advapi32.GetSecurityInfo.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    wintypes.DWORD,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_void_p),
]
_advapi32.GetSecurityInfo.restype = wintypes.DWORD

_advapi32.GetAclInformation.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.c_int,
]
_advapi32.GetAclInformation.restype = wintypes.BOOL

_advapi32.GetAce.argtypes = [ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p)]
_advapi32.GetAce.restype = wintypes.BOOL

_advapi32.EqualSid.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_advapi32.EqualSid.restype = wintypes.BOOL

_advapi32.OpenProcessToken.argtypes = [
    wintypes.HANDLE,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.HANDLE),
]
_advapi32.OpenProcessToken.restype = wintypes.BOOL

_advapi32.GetTokenInformation.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
]
_advapi32.GetTokenInformation.restype = wintypes.BOOL

_kernel32.GetCurrentProcess.restype = wintypes.HANDLE
_kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
_kernel32.LocalFree.argtypes = [ctypes.c_void_p]
_kernel32.ReOpenFile.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD]
_kernel32.ReOpenFile.restype = wintypes.HANDLE
_kernel32.CreateFileW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.c_void_p,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.HANDLE,
]
_kernel32.CreateFileW.restype = wintypes.HANDLE
_kernel32.SetFilePointerEx.argtypes = [
    wintypes.HANDLE,
    ctypes.c_longlong,
    ctypes.POINTER(ctypes.c_longlong),
    wintypes.DWORD,
]
_kernel32.SetFilePointerEx.restype = wintypes.BOOL
_kernel32.SetEndOfFile.argtypes = [wintypes.HANDLE]
_kernel32.SetEndOfFile.restype = wintypes.BOOL


class _ByHandleFileInformation(ctypes.Structure):
    _fields_ = [
        ("dwFileAttributes", wintypes.DWORD),
        ("ftCreationTime", wintypes.FILETIME),
        ("ftLastAccessTime", wintypes.FILETIME),
        ("ftLastWriteTime", wintypes.FILETIME),
        ("dwVolumeSerialNumber", wintypes.DWORD),
        ("nFileSizeHigh", wintypes.DWORD),
        ("nFileSizeLow", wintypes.DWORD),
        ("nNumberOfLinks", wintypes.DWORD),
        ("nFileIndexHigh", wintypes.DWORD),
        ("nFileIndexLow", wintypes.DWORD),
    ]


_kernel32.GetFileInformationByHandle.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(_ByHandleFileInformation),
]
_kernel32.GetFileInformationByHandle.restype = wintypes.BOOL


class _AclSizeInformation(ctypes.Structure):
    _fields_ = [
        ("AceCount", wintypes.DWORD),
        ("AclBytesInUse", wintypes.DWORD),
        ("AclBytesFree", wintypes.DWORD),
    ]


class _AceHeader(ctypes.Structure):
    _fields_ = [
        ("AceType", ctypes.c_ubyte),
        ("AceFlags", ctypes.c_ubyte),
        ("AceSize", wintypes.WORD),
    ]


def _current_user_sid() -> tuple[ctypes.c_void_p, ctypes.Array[ctypes.c_char]]:
    """The calling process's user SID, plus the buffer it lives in.

    Caller must keep the buffer alive for as long as the SID pointer is used
    -- the SID is a view into it, not a copy.
    """
    token = wintypes.HANDLE()
    if not _advapi32.OpenProcessToken(
        _kernel32.GetCurrentProcess(), _TOKEN_QUERY, ctypes.byref(token)
    ):
        raise ctypes.WinError(ctypes.get_last_error())  # type: ignore[attr-defined]
    try:
        size = wintypes.DWORD(0)
        _advapi32.GetTokenInformation(token, _TokenUser, None, 0, ctypes.byref(size))
        buf = ctypes.create_string_buffer(size.value)
        if not _advapi32.GetTokenInformation(token, _TokenUser, buf, size, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())  # type: ignore[attr-defined]
        # TOKEN_USER is { SID_AND_ATTRIBUTES { PSID Sid; DWORD Attributes; } };
        # the first pointer-sized field is the PSID.
        sid = ctypes.cast(buf, ctypes.POINTER(ctypes.c_void_p))[0]
        return ctypes.c_void_p(sid), buf
    finally:
        _kernel32.CloseHandle(token)


def _apply_owner_only_dacl(handle: wintypes.HANDLE) -> bool:
    """Replace *handle*'s DACL with a single owner-only entry.

    *handle* must already carry ``WRITE_DAC``.
    """
    try:
        sid, _buf = _current_user_sid()
    except OSError:
        return False

    trustee = _TRUSTEE_W()
    _advapi32.BuildTrusteeWithSidW(ctypes.byref(trustee), sid)
    entry = _EXPLICIT_ACCESS_W()
    entry.grfAccessPermissions = _FILE_ALL_ACCESS
    entry.grfAccessMode = _SET_ACCESS
    entry.grfInheritance = _NO_INHERITANCE
    entry.Trustee = trustee

    new_dacl = ctypes.c_void_p()
    # Passing NULL for the existing-ACL argument builds a fresh ACL containing
    # only *entry*, rather than merging with whatever inherited entries the
    # file already had -- those inherited entries are exactly what made it
    # readable, so merging would defeat the point.
    if (
        _advapi32.SetEntriesInAclW(1, ctypes.byref(entry), None, ctypes.byref(new_dacl))
        != _ERROR_SUCCESS
    ):
        return False
    try:
        rc = _advapi32.SetSecurityInfo(
            handle,
            _SE_FILE_OBJECT,
            _DACL_SECURITY_INFORMATION | _PROTECTED_DACL_SECURITY_INFORMATION,
            None,
            None,
            new_dacl,
            None,
        )
        return bool(rc == _ERROR_SUCCESS)
    finally:
        _kernel32.LocalFree(new_dacl)


def restrict_handle(fd: int) -> bool:
    """Make the open file behind CRT descriptor *fd* owner-only.

    The CRT's own ``CreateFileW`` call underneath ``os.open``/``open()`` does
    not request ``WRITE_DAC``, so this reopens the same file object (via
    ``ReOpenFile`` -- no path involved, so nothing for a symlink to redirect)
    with that right, applies the DACL, and closes the reopened handle. *fd*
    itself is left exactly as the caller had it.
    """
    try:
        original = wintypes.HANDLE(msvcrt.get_osfhandle(fd))  # type: ignore[attr-defined]
    except OSError:
        return False
    reopened = _kernel32.ReOpenFile(
        original,
        _WRITE_DAC | _READ_CONTROL,
        _FILE_SHARE_READ | _FILE_SHARE_WRITE | _FILE_SHARE_DELETE,
        0,
    )
    if not reopened or reopened == _INVALID_HANDLE_VALUE:
        return False
    try:
        return _apply_owner_only_dacl(reopened)
    finally:
        _kernel32.CloseHandle(reopened)


def restrict_path(path: str) -> bool:
    """Make the file at *path* owner-only without following a symlink there.

    For files this process did not open itself (rotated log backups).
    ``FILE_FLAG_OPEN_REPARSE_POINT`` opens a symlink/junction as the link
    itself rather than its target, the Windows equivalent of the POSIX
    ``O_NOFOLLOW`` the caller's own ``islink`` check already guards against
    on both platforms.
    """
    handle = _kernel32.CreateFileW(
        path,
        _WRITE_DAC | _READ_CONTROL,
        _FILE_SHARE_READ | _FILE_SHARE_WRITE | _FILE_SHARE_DELETE,
        None,
        _OPEN_EXISTING,
        _FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    if not handle or handle == _INVALID_HANDLE_VALUE:
        return False
    try:
        return _apply_owner_only_dacl(handle)
    finally:
        _kernel32.CloseHandle(handle)


def open_no_follow(path: str, *, truncate: bool) -> int:
    """Create-or-open *path* for append/truncate, refusing a symlink or
    junction there, and apply the owner-only DACL -- all against the one
    HANDLE this opens, so there is no window between checking the path and
    writing through it.

    This is the primitive :func:`headroom.fileperms.open_owner_only` needs
    and plain ``os.open()`` cannot provide on Windows: the CRT open it wraps
    has no ``O_NOFOLLOW`` equivalent, so it silently follows a planted
    symlink/junction to whatever it points at, and anything opened that way
    -- including a later DACL change -- lands on the target, not the
    intended file. ``FILE_FLAG_OPEN_REPARSE_POINT`` makes this CreateFileW
    call open the reparse point itself rather than its target regardless of
    *disposition*, so the attribute check below sees the real thing even on
    the create-or-open path a planted link races against.

    Raises ``OSError`` on any failure, including when *path* is a symlink or
    junction -- matching the POSIX ``O_NOFOLLOW`` contract this mirrors, and
    the caller's existing "failing closed is deliberate" policy.

    Always opens with ``OPEN_EXISTING`` first, regardless of *truncate*.
    ``CREATE_ALWAYS``/``OPEN_ALWAYS`` against an existing reparse point does
    not error or open the link -- it deletes it and creates a fresh regular
    file in its place, *before* this function ever gets a handle to inspect,
    so the refusal below would silently never fire for exactly the request
    (truncate) that most needs it. Truncating, when asked for, happens via
    ``SetEndOfFile`` on the one handle this function opens, once that handle
    is confirmed not to be a reparse point -- never by asking ``CreateFileW``
    to recreate the path.
    """
    access = _GENERIC_WRITE | _WRITE_DAC
    share = _FILE_SHARE_READ | _FILE_SHARE_WRITE | _FILE_SHARE_DELETE
    flags = _FILE_FLAG_OPEN_REPARSE_POINT | _FILE_ATTRIBUTE_NORMAL

    handle = _kernel32.CreateFileW(path, access, share, None, _OPEN_EXISTING, flags, None)
    if not handle or handle == _INVALID_HANDLE_VALUE:
        err = ctypes.get_last_error()  # type: ignore[attr-defined]
        if err != _ERROR_FILE_NOT_FOUND:
            raise OSError(f"CreateFileW failed for {path!r}: error {err}")
        # Nothing there yet. CREATE_NEW fails outright instead of silently
        # opening whatever a symlink planted in the gap since the check
        # above left behind -- the same race OPEN_EXISTING already closed
        # for the file-exists case, closed here for the file-missing one.
        handle = _kernel32.CreateFileW(path, access, share, None, _CREATE_NEW, flags, None)
        if not handle or handle == _INVALID_HANDLE_VALUE:
            err = ctypes.get_last_error()  # type: ignore[attr-defined]
            raise OSError(f"CreateFileW failed for {path!r}: error {err}")
    try:
        info = _ByHandleFileInformation()
        if not _kernel32.GetFileInformationByHandle(handle, ctypes.byref(info)):
            err = ctypes.get_last_error()  # type: ignore[attr-defined]
            raise OSError(f"GetFileInformationByHandle failed for {path!r}: error {err}")
        if info.dwFileAttributes & _FILE_ATTRIBUTE_REPARSE_POINT:
            raise OSError(f"refusing to open {path!r}: it is a symlink or junction")
        if truncate:
            if not _kernel32.SetFilePointerEx(handle, 0, None, _FILE_BEGIN):
                err = ctypes.get_last_error()  # type: ignore[attr-defined]
                raise OSError(f"SetFilePointerEx failed for {path!r}: error {err}")
            if not _kernel32.SetEndOfFile(handle):
                err = ctypes.get_last_error()  # type: ignore[attr-defined]
                raise OSError(f"SetEndOfFile failed for {path!r}: error {err}")
        # Best-effort, like every other restrict_* call: a failure here does
        # not fail the open (see fileperms.verify_owner_only for why), but a
        # planted symlink always fails above regardless of this result.
        _apply_owner_only_dacl(handle)
        fd: int = msvcrt.open_osfhandle(  # type: ignore[attr-defined]
            handle, os.O_APPEND if not truncate else os.O_TRUNC
        )
        return fd
    except BaseException:
        _kernel32.CloseHandle(handle)
        raise


def verify_handle(fd: int) -> bool:
    """Like :func:`verify_path`, for a file this process already holds open."""
    try:
        handle = wintypes.HANDLE(msvcrt.get_osfhandle(fd))  # type: ignore[attr-defined]
    except OSError:
        return False
    return _verify_handle(handle)


def verify_path(path: str) -> bool:
    """Re-read the DACL at *path* and confirm it grants access to exactly the
    calling user and no one else.

    Independent of :func:`restrict_path`/:func:`restrict_handle`'s own
    success return: that return says the API calls reported success, this
    re-reads what is actually on the object now. Used to decide whether to
    warn, so the warning reflects the file in front of it rather than a
    static assumption about the platform. Only needs ``READ_CONTROL``, not
    ``WRITE_DAC`` -- it never modifies the DACL.
    """
    handle = _kernel32.CreateFileW(
        path,
        _READ_CONTROL,
        _FILE_SHARE_READ | _FILE_SHARE_WRITE | _FILE_SHARE_DELETE,
        None,
        _OPEN_EXISTING,
        _FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    if not handle or handle == _INVALID_HANDLE_VALUE:
        return False
    try:
        return _verify_handle(handle)
    finally:
        _kernel32.CloseHandle(handle)


def _verify_handle(handle: wintypes.HANDLE) -> bool:
    try:
        sid, _buf = _current_user_sid()
    except OSError:
        return False

    dacl = ctypes.c_void_p()
    sec_desc = ctypes.c_void_p()
    rc = _advapi32.GetSecurityInfo(
        handle,
        _SE_FILE_OBJECT,
        _DACL_SECURITY_INFORMATION,
        None,
        None,
        ctypes.byref(dacl),
        None,
        ctypes.byref(sec_desc),
    )
    if rc != _ERROR_SUCCESS:
        return False
    try:
        size_info = _AclSizeInformation()
        if not _advapi32.GetAclInformation(
            dacl,
            ctypes.byref(size_info),
            ctypes.sizeof(size_info),
            2,  # AclSizeInformation
        ):
            return False
        if size_info.AceCount != 1:
            return False
        ace_ptr = ctypes.c_void_p()
        if not _advapi32.GetAce(dacl, 0, ctypes.byref(ace_ptr)):
            return False
        # ACCESS_ALLOWED_ACE lays out as ACE_HEADER, then a DWORD access mask,
        # then the SID -- the SID starts right after those two fixed fields.
        ace_addr = ace_ptr.value
        if ace_addr is None:
            return False
        sid_offset = ctypes.sizeof(_AceHeader) + ctypes.sizeof(wintypes.DWORD)
        ace_sid = ctypes.c_void_p(ace_addr + sid_offset)
        return bool(_advapi32.EqualSid(ace_sid, sid))
    finally:
        # sec_desc, not dacl: GetSecurityInfo returns pointers into one
        # LocalAlloc'd block: freeing dacl (an offset into it, not its start)
        # is an invalid free. See cpython ctypes issue discussions on
        # GetSecurityInfo for the same gotcha.
        _kernel32.LocalFree(sec_desc)

"""Anchored no-symlink I/O for restricted immutable research artefacts."""

from __future__ import annotations

import fcntl
import hashlib
import os
import secrets
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
_WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
_ALLOWED_DIRECTORY_MODES = {0o700, 0o755}
_ALLOWED_FILE_MODES = {0o600, 0o644}


@dataclass(frozen=True, slots=True)
class AnchoredFile:
    """Bytes and kernel identity observed through an anchored directory handle."""

    content: bytes
    sha256: str
    size: int
    mode: int
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class AnchoredSubprocessFile:
    """A short-lived inherited descriptor reference for one subprocess call."""

    path: str
    pass_fds: tuple[int, ...]


class AnchoredOpenFile:
    """One immutable regular file held open beneath an :class:`AnchoredDirectory`."""

    def __init__(
        self,
        anchor: AnchoredDirectory,
        filename: str,
        file_fd: int,
        observed: os.stat_result,
        content_sha256: str,
    ) -> None:
        self._anchor = anchor
        self.filename = filename
        self._file_fd = file_fd
        self.device = observed.st_dev
        self.inode = observed.st_ino
        self.size = observed.st_size
        self.mtime_ns = observed.st_mtime_ns
        self.mode = stat.S_IMODE(observed.st_mode)
        self.nlink = observed.st_nlink
        self.sha256 = content_sha256
        self._closed = False

    @property
    def name(self) -> str:
        return self.filename

    @property
    def suffix(self) -> str:
        return Path(self.filename).suffix

    def is_file(self) -> bool:
        self.verify_current_file()
        return True

    def __fspath__(self) -> str:
        self._require_open()
        return f"/dev/fd/{self._file_fd}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (str, os.PathLike)):
            return self._anchor.path / self.filename == Path(other)
        return self is other

    __hash__ = None

    def __enter__(self) -> AnchoredOpenFile:
        self._require_open()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __copy__(self) -> None:
        raise TypeError("anchored open files cannot be copied")

    def __deepcopy__(self, _memo: object) -> None:
        raise TypeError("anchored open files cannot be copied")

    def close(self) -> None:
        if not self._closed:
            try:
                self.verify_current_file()
            finally:
                os.close(self._file_fd)
                self._closed = True

    def read_bytes(self) -> bytes:
        """Return the exact held-inode bytes after full pre/post validation."""
        self.verify_current_file()
        content = _pread_all(self._file_fd, self.size)
        if hashlib.sha256(content).hexdigest() != self.sha256:
            raise RuntimeError(f"restricted artefact {self.filename!r} content changed")
        self.verify_current_file()
        return content

    def verify_current_file(self) -> None:
        """Require stable identity, metadata, bytes and anchored lexical identity."""
        self._require_open()
        self._anchor.verify_path_identity()
        observed = os.fstat(self._file_fd)
        identity = (
            observed.st_dev,
            observed.st_ino,
            observed.st_size,
            observed.st_mtime_ns,
            stat.S_IMODE(observed.st_mode),
            observed.st_nlink,
        )
        expected = (
            self.device,
            self.inode,
            self.size,
            self.mtime_ns,
            self.mode,
            self.nlink,
        )
        if identity != expected or not stat.S_ISREG(observed.st_mode):
            raise RuntimeError(f"restricted artefact {self.filename!r} identity changed")
        content = _pread_all(self._file_fd, self.size)
        if hashlib.sha256(content).hexdigest() != self.sha256:
            raise RuntimeError(f"restricted artefact {self.filename!r} content changed")
        self._anchor.verify_path_identity()

    @contextmanager
    def duplicate_for_subprocess(self) -> Iterator[AnchoredSubprocessFile]:
        """Yield an inheritable duplicate addressable only for one subprocess call."""
        self.verify_current_file()
        duplicate = os.dup(self._file_fd)
        try:
            os.set_inheritable(duplicate, True)
            yield AnchoredSubprocessFile(
                path=f"/dev/fd/{duplicate}", pass_fds=(duplicate,)
            )
        finally:
            os.close(duplicate)
            self.verify_current_file()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("anchored file is closed")


class AnchoredDirectory:
    """A directory held by fd and continuously bound to its lexical path identity."""

    def __init__(
        self, path: Path, directory_fd: int, device: int, inode: int, mode: int
    ) -> None:
        self.path = path
        self._directory_fd = directory_fd
        self.device = device
        self.inode = inode
        self.mode = mode
        self._closed = False
        self._lock_depth = 0

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        create: bool = False,
        directory_mode: int | None = 0o700,
    ) -> AnchoredDirectory:
        """Open ``path`` without following any component symlink.

        When ``create`` is true, missing components are created one at a time
        relative to the already-open parent. Existing components must all be real
        directories. The returned fd anchors subsequent operations even if an
        attacker renames a path component.
        """
        if directory_mode is not None and directory_mode not in _ALLOWED_DIRECTORY_MODES:
            raise ValueError("anchored directory mode must be 0700 or 0755")
        if create and directory_mode is None:
            raise ValueError("created anchored directories require an exact mode")
        absolute = _lexical_absolute(path)
        directory_fd, observed = _open_directory_path(
            absolute,
            create=create,
            directory_mode=directory_mode if directory_mode is not None else 0o700,
        )
        observed_mode = stat.S_IMODE(observed.st_mode)
        if observed_mode not in _ALLOWED_DIRECTORY_MODES or (
            directory_mode is not None and observed_mode != directory_mode
        ):
            os.close(directory_fd)
            raise ValueError(
                "anchored directory mode must be "
                + (
                    f"{directory_mode:04o}"
                    if directory_mode is not None
                    else "0700 or 0755"
                )
            )
        return cls(
            absolute,
            directory_fd,
            observed.st_dev,
            observed.st_ino,
            observed_mode,
        )

    def __enter__(self) -> AnchoredDirectory:
        self._require_open()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            os.close(self._directory_fd)
            self._closed = True

    def __del__(self) -> None:
        if not getattr(self, "_closed", True):
            self.close()

    def verify_path_identity(self) -> None:
        """Rewalk every component and require the path still names this directory."""
        self._require_open()
        check_fd, observed = _open_directory_path(
            self.path, create=False, directory_mode=0o700
        )
        try:
            current = os.fstat(self._directory_fd)
            if (
                not stat.S_ISDIR(current.st_mode)
                or (current.st_dev, current.st_ino) != (self.device, self.inode)
                or (observed.st_dev, observed.st_ino) != (self.device, self.inode)
                or stat.S_IMODE(current.st_mode) != self.mode
                or stat.S_IMODE(observed.st_mode) != self.mode
            ):
                raise RuntimeError("anchored directory path identity changed")
        finally:
            os.close(check_fd)

    def list_names(self) -> tuple[str, ...]:
        """List anchored directory entries after pre/post identity checks."""
        self.verify_path_identity()
        names = tuple(sorted(os.listdir(self._directory_fd)))
        self.verify_path_identity()
        return names

    @contextmanager
    def exclusive_lock(self) -> Iterator[None]:
        """Serialize cooperating bundle publishers; kernel release is crash-safe."""
        self._require_open()
        if self._lock_depth == 0:
            fcntl.flock(self._directory_fd, fcntl.LOCK_EX)
        self._lock_depth += 1
        try:
            self.verify_path_identity()
            yield
            self.verify_path_identity()
        finally:
            self._lock_depth -= 1
            if self._lock_depth == 0:
                fcntl.flock(self._directory_fd, fcntl.LOCK_UN)

    def read_bytes(
        self, filename: str, *, expected_mode: int | None = 0o600
    ) -> AnchoredFile:
        """Read one regular basename with no symlink traversal and stable fstat checks."""
        with self.open_file(filename, expected_mode=expected_mode) as opened:
            content = opened.read_bytes()
            return AnchoredFile(
                content=content,
                sha256=opened.sha256,
                size=opened.size,
                mode=opened.mode,
                device=opened.device,
                inode=opened.inode,
            )

    def verify_file(
        self,
        filename: str,
        observed: AnchoredFile,
        *,
        expected_mode: int | None = 0o600,
    ) -> None:
        """Require a basename still to identify the exact bytes and inode read earlier."""
        if not isinstance(observed, AnchoredFile):
            raise TypeError("observed file must be an AnchoredFile")
        current = self.read_bytes(filename, expected_mode=expected_mode)
        if current != observed:
            raise RuntimeError(
                f"restricted artefact {filename!r} changed after validation"
            )

    def open_file(
        self,
        filename: str,
        *,
        expected_mode: int | None = 0o600,
        expected_sha256: str | None = None,
    ) -> AnchoredOpenFile:
        """Open and hash one stable basename while retaining its exact descriptor."""
        name = _safe_basename(filename)
        if expected_mode is not None and expected_mode not in _ALLOWED_FILE_MODES:
            raise ValueError("anchored file mode must be 0600 or 0644")
        if expected_sha256 is not None and (
            len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
        self.verify_path_identity()
        try:
            file_fd = os.open(name, _READ_FLAGS, dir_fd=self._directory_fd)
        except OSError as error:
            raise ValueError(f"restricted artefact {name!r} is unavailable or unsafe") from error
        try:
            before = os.fstat(file_fd)
            _validate_regular_mode(before, expected_mode, name)
            content = _pread_all(file_fd, before.st_size)
            after = os.fstat(file_fd)
            before_identity = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                stat.S_IMODE(before.st_mode),
                before.st_nlink,
            )
            after_identity = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                stat.S_IMODE(after.st_mode),
                after.st_nlink,
            )
            if before_identity != after_identity:
                raise RuntimeError(f"restricted artefact {name!r} changed during read")
            digest = hashlib.sha256(content).hexdigest()
            if expected_sha256 is not None and digest != expected_sha256:
                raise ValueError(f"restricted artefact {name!r} checksum mismatch")
            opened = AnchoredOpenFile(self, name, file_fd, after, digest)
        except Exception:
            os.close(file_fd)
            raise
        self.verify_path_identity()
        return opened

    def publish_bytes(
        self,
        filename: str,
        content: bytes,
        *,
        mode: int = 0o600,
        allow_existing: bool = True,
    ) -> AnchoredFile:
        """Atomically publish complete bytes, optionally reusing an identical file."""
        name = _safe_basename(filename)
        if not isinstance(content, bytes):
            raise TypeError("restricted artefact content must be bytes")
        if mode not in _ALLOWED_FILE_MODES:
            raise ValueError("anchored file mode must be 0600 or 0644")
        self.verify_path_identity()
        temporary_name, file_fd = self._create_temporary_file(mode)
        created_identity: tuple[int, int] | None = None
        linked = False
        try:
            try:
                os.fchmod(file_fd, mode)
                view = memoryview(content)
                offset = 0
                while offset < len(view):
                    written = os.write(file_fd, view[offset:])
                    if written <= 0:
                        raise RuntimeError(
                            f"short write for restricted artefact {name!r}"
                        )
                    offset += written
                os.fsync(file_fd)
                observed = os.fstat(file_fd)
                _validate_regular_mode(observed, mode, name)
                if observed.st_size != len(content):
                    raise RuntimeError(
                        f"restricted artefact {name!r} write size drifted"
                    )
                created_identity = (observed.st_dev, observed.st_ino)
            finally:
                os.close(file_fd)
            try:
                os.link(
                    temporary_name,
                    name,
                    src_dir_fd=self._directory_fd,
                    dst_dir_fd=self._directory_fd,
                    follow_symlinks=False,
                )
                linked = True
            except FileExistsError:
                if not allow_existing:
                    raise FileExistsError(
                        f"restricted artefact {name!r} already exists"
                    ) from None
                existing = self.read_bytes(name, expected_mode=mode)
                if existing.content != content:
                    raise FileExistsError(
                        f"refusing to replace restricted artefact {name!r}"
                    ) from None
                return existing
            except OSError as error:
                raise ValueError(
                    f"cannot publish restricted artefact {name!r} safely"
                ) from error
            os.fsync(self._directory_fd)
        finally:
            try:
                os.unlink(temporary_name, dir_fd=self._directory_fd)
            except FileNotFoundError:
                pass
            os.fsync(self._directory_fd)
        if not linked:
            raise RuntimeError(f"restricted artefact {name!r} was not published")
        reopened = self.read_bytes(name, expected_mode=mode)
        if (
            created_identity != (reopened.device, reopened.inode)
            or reopened.content != content
            or reopened.sha256 != hashlib.sha256(content).hexdigest()
        ):
            raise RuntimeError(f"restricted artefact {name!r} failed post-write verification")
        self.verify_path_identity()
        return reopened

    def _create_temporary_file(self, mode: int) -> tuple[str, int]:
        for _attempt in range(32):
            temporary_name = f".publish-{secrets.token_hex(16)}.tmp"
            try:
                return temporary_name, os.open(
                    temporary_name,
                    _WRITE_FLAGS,
                    mode,
                    dir_fd=self._directory_fd,
                )
            except FileExistsError:
                continue
            except OSError as error:
                raise ValueError("cannot create atomic publication temporary file") from error
        raise RuntimeError("cannot allocate atomic publication temporary file")

    def replace_bytes(
        self,
        filename: str,
        content: bytes,
        *,
        mode: int = 0o600,
    ) -> AnchoredFile:
        """Atomically replace one mutable file without exposing partial bytes."""
        name = _safe_basename(filename)
        if not isinstance(content, bytes):
            raise TypeError("restricted artefact content must be bytes")
        if mode not in _ALLOWED_FILE_MODES:
            raise ValueError("anchored file mode must be 0600 or 0644")
        self.verify_path_identity()
        if name in self.list_names():
            self.read_bytes(name, expected_mode=mode)
        temporary_name, file_fd = self._create_temporary_file(mode)
        created_identity: tuple[int, int] | None = None
        replaced = False
        try:
            try:
                os.fchmod(file_fd, mode)
                view = memoryview(content)
                offset = 0
                while offset < len(view):
                    written = os.write(file_fd, view[offset:])
                    if written <= 0:
                        raise RuntimeError(
                            f"short write for restricted artefact {name!r}"
                        )
                    offset += written
                os.fsync(file_fd)
                observed = os.fstat(file_fd)
                _validate_regular_mode(observed, mode, name)
                if observed.st_size != len(content):
                    raise RuntimeError(
                        f"restricted artefact {name!r} write size drifted"
                    )
                created_identity = (observed.st_dev, observed.st_ino)
            finally:
                os.close(file_fd)
            self.verify_path_identity()
            os.replace(
                temporary_name,
                name,
                src_dir_fd=self._directory_fd,
                dst_dir_fd=self._directory_fd,
            )
            replaced = True
            os.fsync(self._directory_fd)
        finally:
            if not replaced:
                try:
                    os.unlink(temporary_name, dir_fd=self._directory_fd)
                except FileNotFoundError:
                    pass
                os.fsync(self._directory_fd)
        reopened = self.read_bytes(name, expected_mode=mode)
        if (
            created_identity != (reopened.device, reopened.inode)
            or reopened.content != content
            or reopened.sha256 != hashlib.sha256(content).hexdigest()
        ):
            raise RuntimeError(f"restricted artefact {name!r} failed replacement verification")
        self.verify_path_identity()
        return reopened

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("anchored directory is closed")


def _lexical_absolute(path: str | Path) -> Path:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute():
        supplied = Path.cwd() / supplied
    if any(part in {".", ".."} for part in supplied.parts[1:]):
        raise ValueError("restricted directory path may not contain dot components")
    return supplied


def _open_directory_path(
    path: Path, *, create: bool, directory_mode: int
) -> tuple[int, os.stat_result]:
    current_fd = os.open("/", _DIRECTORY_FLAGS)
    try:
        for component in path.parts[1:]:
            try:
                lexical = os.stat(component, dir_fd=current_fd, follow_symlinks=False)
            except FileNotFoundError:
                if not create:
                    raise ValueError(
                        f"restricted directory component {component!r} is unavailable"
                    ) from None
                try:
                    os.mkdir(component, directory_mode, dir_fd=current_fd)
                    os.fsync(current_fd)
                except FileExistsError:
                    pass
                lexical = os.stat(component, dir_fd=current_fd, follow_symlinks=False)
            if not stat.S_ISDIR(lexical.st_mode):
                raise ValueError(
                    f"restricted directory component {component!r} is not a real directory"
                )
            try:
                next_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=current_fd)
            except OSError as error:
                raise ValueError(
                    f"restricted directory component {component!r} is unsafe"
                ) from error
            opened = os.fstat(next_fd)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != (lexical.st_dev, lexical.st_ino)
            ):
                os.close(next_fd)
                raise RuntimeError("restricted directory component changed while opening")
            os.close(current_fd)
            current_fd = next_fd
        final = os.fstat(current_fd)
        return current_fd, final
    except Exception:
        os.close(current_fd)
        raise


def _safe_basename(filename: str) -> str:
    if (
        not isinstance(filename, str)
        or not filename
        or filename in {".", ".."}
        or Path(filename).name != filename
        or "/" in filename
        or "\x00" in filename
    ):
        raise ValueError("restricted artefact filename must be a safe basename")
    return filename


def _validate_regular_mode(
    observed: os.stat_result, expected_mode: int | None, name: str
) -> None:
    if not stat.S_ISREG(observed.st_mode):
        raise ValueError(f"restricted artefact {name!r} is not a regular file")
    if observed.st_nlink != 1:
        raise ValueError(f"restricted artefact {name!r} must have exactly one link")
    observed_mode = stat.S_IMODE(observed.st_mode)
    if observed_mode not in _ALLOWED_FILE_MODES or (
        expected_mode is not None and observed_mode != expected_mode
    ):
        required_mode = (
            f"{expected_mode:04o}"
            if expected_mode is not None
            else "0600 or 0644"
        )
        raise ValueError(
            f"restricted artefact {name!r} mode must be {required_mode}"
        )


def _pread_all(file_fd: int, size: int) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        chunk = os.pread(file_fd, min(1024 * 1024, size - offset), offset)
        if not chunk:
            break
        chunks.append(chunk)
        offset += len(chunk)
    content = b"".join(chunks)
    if len(content) != size:
        raise RuntimeError("restricted artefact size changed during read")
    return content

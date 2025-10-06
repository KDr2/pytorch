"""Cache implementation classes for PyTorch Inductor runtime caching.

This module provides concrete implementations of caching backends including
in-memory, on-disk, and remote caching strategies. Each implementation follows
the abstract _CacheImpl interface and provides thread-safe operations with
appropriate locking mechanisms.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from io import BufferedReader, BufferedWriter
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Generator, Optional, override, Self

from filelock import FileLock

from . import locks, utils


@dataclass
class Get:
    """Result wrapper for cache get operations.
    
    Allows distinguishing between a cache miss and a cached None value.
    
    Attributes:
        hit: True if the key was found in the cache, False otherwise.
        value: The cached value if hit is True, None otherwise.
    """
    hit: bool
    value: Any = None


@dataclass
class Insert:
    """Result wrapper for cache insert operations.
    
    Attributes:
        inserted: True if the key-value pair was successfully inserted,
                 False if the key already existed in the cache.
    """
    inserted: bool


class _CacheImpl(ABC):
    """Abstract base class for cache implementations.
    
    This class defines the interface that all cache implementations must follow.
    It provides thread-safe operations through a locking mechanism and supports
    both get and insert operations.
    
    Note: We don't use generics here as doing so would require that the interfaces
    know which k/v types the implementation can work with. Instead, we leave that
    determination up to the implementation itself and require that the interfaces
    handle any potential errors from invalid k/v types being passed to the
    implementation.
    """
    
    def __init__(self: Self) -> None:
        """Initialize the cache implementation with a threading lock."""
        self._lock: Lock = Lock()
    
    @property
    def lock(self: Self) -> Callable[[int], Generator[None, None, None]]:
        """Get a context manager for acquiring the cache lock.
        
        Locking of the cache is not done by the implementation itself, but by the
        interface that uses it. The interface may want to hold the lock for longer
        than a single cache operation, for example when dealing with multiple
        cache implementations at once, so we leave that decision up to the interface.
        
        Args:
            timeout: Timeout in seconds for acquiring the lock.
            
        Returns:
            A callable that returns a context manager for the lock.
        """
        return lambda timeout=None: locks._acquire_lock_with_timeout(self._lock, timeout) 

    @abstractmethod
    def get(self: Self, key: Any) -> Get:
        """Retrieve a value from the cache.
        
        Args:
            key: The key to look up in the cache.
            
        Returns:
            A Get object indicating whether the key was found and its value.
        """
        pass

    @abstractmethod
    def insert(self: Self, key: Any, value: Any) -> Insert:
        """Insert a key-value pair into the cache.
        
        Args:
            key: The key to insert.
            value: The value to associate with the key.
            
        Returns:
            An Insert object indicating whether the insertion was successful.
        """
        pass


class _InMemoryCacheImpl(_CacheImpl):
    """In-memory cache implementation using a dictionary.
    
    This implementation stores key-value pairs in a Python dictionary,
    with keys being pickled for consistent hashing. It provides fast
    access but is limited by available memory and process lifetime.
    """

    def __init__(self: Self) -> None:
        """Initialize the in-memory cache with an empty dictionary."""
        super().__init__()
        self._memory: dict[bytes, Any] = {}

    @override
    def get(self: Self, key: Any) -> Get:
        """Retrieve a value from the in-memory cache.
        
        Args:
            key: The key to look up. Will be pickled for storage.
            
        Returns:
            A Get object with hit=True and the value if found,
            or hit=False if not found.
        """
        pickled_key: bytes = utils._try_pickle_key(key)
        if (value := self._memory.get(pickled_key)):
            return Get(hit=True, value=value)
        return Get(hit=False)

    @override
    def insert(self: Self, key: Any, value: Any) -> Insert:
        """Insert a key-value pair into the in-memory cache.
        
        Args:
            key: The key to insert. Will be pickled for storage.
            value: The value to associate with the key.
            
        Returns:
            An Insert object with inserted=True if the key was new,
            or inserted=False if the key already existed.
        """
        pickled_key: bytes = utils._try_pickle_key(key)
        if pickled_key not in self._memory:
            self._memory[pickled_key] = value
            return Insert(inserted=True)
        return Insert(inserted=False)


class _OnDiskCacheImpl(_CacheImpl):
    """On-disk cache implementation using file system storage.
    
    This implementation stores cached data as files on disk, with version
    headers to handle cache invalidation. It uses file locking to ensure
    thread safety across processes and provides persistent storage that
    survives process restarts.
    
    Attributes:
        _version: Version number for cache format compatibility.
        _version_header_length: Length of the version header in bytes.
    """
    _version: int = 0
    _version_header_length: int = 4

    def __init__(self: Self, sub_dir: Optional[str] = None) -> None:
        """Initialize the on-disk cache with a specified subdirectory.
        
        Args:
            sub_dir: Subdirectory name within the cache directory.
                    Defaults to empty string if not specified.
        """
        self._sub_dir: str = sub_dir
        self._flock: FileLock = FileLock(str(self._base_dir / "dir.lock"))
    
    @property
    def _base_dir(self: Self) -> Path:
        """Get the base directory for cache storage.
        
        Returns:
            Path to the cache directory based on the default cache dir
            and the specified subdirectory.
        """
        from torch._inductor.runtime.runtime_utils import default_cache_dir

        return Path(default_cache_dir(), "cache", self._sub_dir or "")
    
    def _fpath_from_key(self: Self, key: Any) -> Path:
        """Generate a file path from a cache key.
        
        Args:
            key: The cache key to convert to a file path.
            
        Returns:
            A Path object representing the file location for this key.
        """
        pickled_key: bytes = utils._try_pickle_key(key)
        return self._base_dir / sha256(pickled_key).hexdigest()[:32]
    
    @classmethod
    def _version_header(cls: Self) -> bytes:
        """Generate the version header bytes.
        
        Returns:
            A byte string representing the current cache version header.
        """
        return sha256(str(cls._version).encode()).digest()[:cls._version_header_length]
    
    def _version_header_matches(self: Self, fp: BufferedReader) -> bool:
        """Check if the file's version header matches the current version.
        
        Args:
            fp: File pointer positioned at the start of the file.
            
        Returns:
            True if the version header matches, False otherwise.
        """
        return fp.read(self._version_header_length) == self._version_header()

    def _write_version_header(self: Self, fp: BufferedWriter) -> None:
        """Write the version header to a file.
        
        Args:
            fp: File pointer where the version header should be written.
        """
        fp.write(self._version_header())

    @override
    @property
    def lock(self: Self) -> Callable[[int], Generator[None, None, None]]:
        """Get a context manager for acquiring the file lock.
        
        Uses file locking to ensure thread safety across processes.
        
        Args:
            timeout: Timeout in seconds for acquiring the file lock.
            
        Returns:
            A callable that returns a context manager for the file lock.
        """
        return lambda timeout=None: locks._acquire_flock_with_timeout(self._flock, timeout) 

    @override
    def get(self: Self, key: Any) -> Get:
        """Retrieve a value from the on-disk cache.
        
        Args:
            key: The key to look up in the cache.
            
        Returns:
            A Get object with hit=True and the value if found,
            or hit=False if not found or version mismatch.
        """
        fpath: Path = self._fpath_from_key(key)

        if not fpath.is_file():
            return Get(hit=False)
        
        pickled_value: Optional[bytes] = None
        with open(fpath, "rb") as fp:
            if self._version_header_matches(fp):
                pickled_value = fp.read()
        
        if not pickled_value:
            # if pickled_value is still None, even though the file exists, then
            # we know that the version header did not match. in this case implementation
            # is up to preference, we choose to remove entries that do not match
            # the version header so that the key can be re-cached later with the correct
            # version header
            fpath.unlink()
            return Get(hit=False)
        
        return Get(hit=True, value=utils._try_unpickle_value(pickled_value))
    
    @override
    def insert(self: Self, key: Any, value: Any) -> Insert:
        """Insert a key-value pair into the on-disk cache.
        
        Args:
            key: The key to insert.
            value: The value to associate with the key.
            
        Returns:
            An Insert object with inserted=True if successfully inserted,
            or inserted=False if the key already exists with a valid version.
        """
        fpath: Path = self._fpath_from_key(key)
        fpath.parent.mkdir(parents=True, exist_ok=True)

        fp: Optional[BufferedWriter] = None
        try:
            fp = open(fpath, "xb")
        except FileExistsError:
            is_stale: bool = False
            with open(fpath, "rb") as fp:
                is_stale = not self._version_header_matches(fp)
            
            if is_stale:
                # same story as above, in this case the version header doesn't
                # match so we choose to remove the old entry so that the new
                # k/v pair can be cached
                fpath.unlink()
                fp = open(fpath, "xb")
            else:
                fp = None
        finally:
            if fp:
                try:
                    pickled_value: bytes = utils._try_pickle_value(value)
                    self._write_version_header(fp)
                    fp.write(pickled_value)
                    return Insert(inserted=True)
                finally:
                    fp.close()
            return Insert(inserted=False)


try:
    from .fb.implementations import _RemoteCacheImpl
except ModuleNotFoundError:
    class _RemoteCacheImpl(_CacheImpl):
        """Fallback remote cache implementation for non-Facebook environments.
        
        This is a no-op implementation that always raises NotImplementedError.
        The actual remote cache implementation is provided in the `.fb` module
        for Facebook-specific environments.
        
        Attributes:
            _version: Version number for cache format compatibility.
            has_strong_consistency: Whether the remote cache provides strong
                                   consistency guarantees.
        """
        _version: int = 0
        has_strong_consistency: bool = False

        def __init__(self: Self) -> None:
            """Initialize the fallback remote cache implementation.
            
            Note: We don't need to initialize any form of lock since this
            implementation provides a pseudo-lock context manager.
            """

        @override
        @property
        def lock(self: Self) -> Callable[[int], Generator[None, None, None]]:
            """Get a pseudo lock that does nothing.
            
            Most remote cache implementations don't have an ability to implement
            any form of locking, so we provide a no-op pseudo-lock for consistency
            with the interface.
            
            Args:
                timeout: Timeout parameter (ignored in this implementation).
                
            Returns:
                A callable that returns a no-op context manager.
            """
            @contextmanager
            def pseudo_lock(timeout: Optional[int] = None) -> Generator[None, None, None]:
                yield
            return pseudo_lock

        @override
        def get(self: Self, key: Any) -> Get:
            """Raise NotImplementedError for remote cache get operations.
            
            Args:
                key: The key to look up (ignored).
                
            Raises:
                NotImplementedError: Always raised as this is a fallback implementation.
            """
            raise NotImplementedError
        
        @override
        def insert(self: Self, key: Any, value: Any) -> Insert:
            """Raise NotImplementedError for remote cache insert operations.
            
            Args:
                key: The key to insert (ignored).
                value: The value to insert (ignored).
                
            Raises:
                NotImplementedError: Always raised as this is a fallback implementation.
            """
            raise NotImplementedError

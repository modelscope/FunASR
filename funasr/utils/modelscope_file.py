# Copyright (c) Alibaba, Inc. and its affiliates.

import contextlib
import os
import tempfile
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Generator, Union

import requests


class Storage(metaclass=ABCMeta):
    """Abstract class of storage.

    All backends need to implement two apis: ``read()`` and ``read_text()``.
    ``read()`` reads the file as a byte stream and ``read_text()`` reads
    the file as texts.
    """

    @abstractmethod
    def read(self, filepath: str):
        pass

    @abstractmethod
    def read_text(self, filepath: str):
        pass

    @abstractmethod
    def write(self, obj: bytes, filepath: Union[str, Path]) -> None:
        pass

    @abstractmethod
    def write_text(self,
                   obj: str,
                   filepath: Union[str, Path],
                   encoding: str = 'utf-8') -> None:
        pass


class LocalStorage(Storage):
    """Local hard disk storage"""

    def read(self, filepath: Union[str, Path]) -> bytes:
        """Read data from a given ``filepath`` with 'rb' mode.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            bytes: Expected bytes object.
        """
        with open(filepath, 'rb') as f:
            content = f.read()
        return content

    def read_text(self,
                  filepath: Union[str, Path],
                  encoding: str = 'utf-8') -> str:
        """Read data from a given ``filepath`` with 'r' mode.

        Args:
            filepath (str or Path): Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Default: 'utf-8'.

        Returns:
            str: Expected text reading from ``filepath``.
        """
        with open(filepath, 'r', encoding=encoding) as f:
            value_buf = f.read()
        return value_buf

    def write(self, obj: bytes, filepath: Union[str, Path]) -> None:
        """Write data to a given ``filepath`` with 'wb' mode.

        Note:
            ``write`` will create a directory if the directory of ``filepath``
            does not exist.

        Args:
            obj (bytes): Data to be written.
            filepath (str or Path): Path to write data.
        """
        dirname = os.path.dirname(filepath)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        with open(filepath, 'wb') as f:
            f.write(obj)

    def write_text(self,
                   obj: str,
                   filepath: Union[str, Path],
                   encoding: str = 'utf-8') -> None:
        """Write data to a given ``filepath`` with 'w' mode.

        Note:
            ``write_text`` will create a directory if the directory of
            ``filepath`` does not exist.

        Args:
            obj (str): Data to be written.
            filepath (str or Path): Path to write data.
            encoding (str): The encoding format used to open the ``filepath``.
                Default: 'utf-8'.
        """
        dirname = os.path.dirname(filepath)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        with open(filepath, 'w', encoding=encoding) as f:
            f.write(obj)

    @contextlib.contextmanager
    def as_local_path(
            self,
            filepath: Union[str,
                            Path]) -> Generator[Union[str, Path], None, None]:
        """Only for unified API and do nothing."""
        yield filepath


class HTTPStorage(Storage):
    """HTTP and HTTPS storage."""

    def read(self, url):
        # TODO @wenmeng.zwm add progress bar if file is too large
        r = requests.get(url)
        r.raise_for_status()
        return r.content

    def read_text(self, url):
        r = requests.get(url)
        r.raise_for_status()
        return r.text

    @contextlib.contextmanager
    def as_local_path(
            self, filepath: str) -> Generator[Union[str, Path], None, None]:
        """Download a file from ``filepath``.

        ``as_local_path`` is decorated by :meth:`contextlib.contextmanager`. It
        can be called with ``with`` statement, and when exists from the
        ``with`` statement, the temporary path will be released.

        Args:
            filepath (str): Download a file from ``filepath``.

        Examples:
            >>> storage = HTTPStorage()
            >>> # After existing from the ``with`` clause,
            >>> # the path will be removed
            >>> with storage.get_local_path('http://path/to/file') as path:
            ...     # do something here
        """
        try:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.write(self.read(filepath))
            f.close()
            yield f.name
        finally:
            os.remove(f.name)

    def write(self, obj: bytes, url: Union[str, Path]) -> None:
        raise NotImplementedError('write is not supported by HTTP Storage')

    def write_text(self,
                   obj: str,
                   url: Union[str, Path],
                   encoding: str = 'utf-8') -> None:
        raise NotImplementedError(
            'write_text is not supported by HTTP Storage')


class OSSStorage(Storage):
    """OSS storage."""

    def __init__(self, oss_config_file=None):
        # read from config file or env var
        raise NotImplementedError(
            'OSSStorage.__init__ to be implemented in the future')

    def read(self, filepath):
        raise NotImplementedError(
            'OSSStorage.read to be implemented in the future')

    def read_text(self, filepath, encoding='utf-8'):
        raise NotImplementedError(
            'OSSStorage.read_text to be implemented in the future')

    @contextlib.contextmanager
    def as_local_path(
            self, filepath: str) -> Generator[Union[str, Path], None, None]:
        """Download a file from ``filepath``.

        ``as_local_path`` is decorated by :meth:`contextlib.contextmanager`. It
        can be called with ``with`` statement, and when exists from the
        ``with`` statement, the temporary path will be released.

        Args:
            filepath (str): Download a file from ``filepath``.

        Examples:
            >>> storage = OSSStorage()
            >>> # After existing from the ``with`` clause,
            >>> # the path will be removed
            >>> with storage.get_local_path('http://path/to/file') as path:
            ...     # do something here
        """
        try:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.write(self.read(filepath))
            f.close()
            yield f.name
        finally:
            os.remove(f.name)

    def write(self, obj: bytes, filepath: Union[str, Path]) -> None:
        raise NotImplementedError(
            'OSSStorage.write to be implemented in the future')

    def write_text(self,
                   obj: str,
                   filepath: Union[str, Path],
                   encoding: str = 'utf-8') -> None:
        raise NotImplementedError(
            'OSSStorage.write_text to be implemented in the future')


G_STORAGES = {}


class File(object):
    _prefix_to_storage: dict = {
        'oss': OSSStorage,
        'http': HTTPStorage,
        'https': HTTPStorage,
        'local': LocalStorage,
    }

    @staticmethod
    def _get_storage(uri):
        assert isinstance(uri,
                          str), f'uri should be str type, but got {type(uri)}'

        if '://' not in uri:
            # local path
            storage_type = 'local'
        else:
            prefix, _ = uri.split('://')
            storage_type = prefix

        assert storage_type in File._prefix_to_storage, \
            f'Unsupported uri {uri}, valid prefixs: '\
            f'{list(File._prefix_to_storage.keys())}'

        if storage_type not in G_STORAGES:
            G_STORAGES[storage_type] = File._prefix_to_storage[storage_type]()

        return G_STORAGES[storage_type]

    @staticmethod
    def read(uri: str) -> bytes:
        """Read data from a given ``filepath`` with 'rb' mode.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            bytes: Expected bytes object.
        """
        storage = File._get_storage(uri)
        return storage.read(uri)

    @staticmethod
    def read_text(uri: Union[str, Path], encoding: str = 'utf-8') -> str:
        """Read data from a given ``filepath`` with 'r' mode.

        Args:
            filepath (str or Path): Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Default: 'utf-8'.

        Returns:
            str: Expected text reading from ``filepath``.
        """
        storage = File._get_storage(uri)
        return storage.read_text(uri)

    @staticmethod
    def write(obj: bytes, uri: Union[str, Path]) -> None:
        """Write data to a given ``filepath`` with 'wb' mode.

        Note:
            ``write`` will create a directory if the directory of ``filepath``
            does not exist.

        Args:
            obj (bytes): Data to be written.
            filepath (str or Path): Path to write data.
        """
        storage = File._get_storage(uri)
        return storage.write(obj, uri)

    @staticmethod
    def write_text(obj: str, uri: str, encoding: str = 'utf-8') -> None:
        """Write data to a given ``filepath`` with 'w' mode.

        Note:
            ``write_text`` will create a directory if the directory of
            ``filepath`` does not exist.

        Args:
            obj (str): Data to be written.
            filepath (str or Path): Path to write data.
            encoding (str): The encoding format used to open the ``filepath``.
                Default: 'utf-8'.
        """
        storage = File._get_storage(uri)
        return storage.write_text(obj, uri)

    @contextlib.contextmanager
    def as_local_path(uri: str) -> Generator[Union[str, Path], None, None]:
        """Only for unified API and do nothing."""
        storage = File._get_storage(uri)
        with storage.as_local_path(uri) as local_path:
            yield local_path

"""Namespace aliases for artifacts written before the source release."""

import hashlib


CURRENT_NAMESPACE = "safe-fcl."
_ARCHIVED_NAMESPACE_SHA256 = "8e8a5447f47aef745be06544c0c77cd8224cd0bc2f4febaf63150495703631ec"


def matches_schema(value: object, current: str) -> bool:
    """Validate the original schema without rewriting the artifact."""
    if not isinstance(value, str):
        return False
    if value == current:
        return True
    namespace, separator, suffix = value.partition(".")
    return (
        bool(separator)
        and suffix == current.removeprefix(CURRENT_NAMESPACE)
        and namespace.isascii()
        and hashlib.sha256(namespace.encode("ascii")).hexdigest() == _ARCHIVED_NAMESPACE_SHA256
    )

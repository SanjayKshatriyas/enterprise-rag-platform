"""Domain-specific exceptions. The API layer maps these to HTTP status codes."""


class RagError(Exception):
    """Base class for all application errors."""


class UnsupportedFileTypeError(RagError):
    """Uploaded file extension is not supported."""


class EmptyDocumentError(RagError):
    """No extractable text was found in the document."""


class DocumentParsingError(RagError):
    """The file is corrupt, encrypted, or otherwise unreadable."""


class ProviderError(RagError):
    """An upstream embedding / LLM provider failed or is misconfigured."""

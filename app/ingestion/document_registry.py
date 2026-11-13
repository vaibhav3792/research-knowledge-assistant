
class DocumentRegistry:
    """Track document identities during the current application process."""

    def __init__(self) -> None:
        self._document_ids: set[str] = set()

    def is_registered(self, document_id: str) -> bool:
        """Check whether a document ID has already been registered."""

        return document_id in self._document_ids

    def register(self, document_id: str) -> bool:
        """Register a document ID.

        Returns True if newly registered, False if already present.
        """

        if self.is_registered(document_id):
            return False

        self._document_ids.add(document_id)
        return True

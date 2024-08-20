class EmptyFileListException(Exception):
    """Exception raised when a list of files is empty."""

    def __init__(self, message="No files left after filtering"):
        self.message = message
        super().__init__(self.message)

from abc import ABC, abstractmethod
from typing import Any, TypeVar

T = TypeVar("T")  # Generic type for the return value


class BaseRetriever(ABC):
    """Abstract base class for all retrievers.

    This class defines the basic interface that all retrievers must implement.
    Each retriever must provide an _execute method that handles the actual
    data retrieval and processing.

    Methods
    -------
    __call__()
        Executes the retriever's main functionality
    _execute()
        Abstract method that must be implemented by subclasses
    """

    def __init__(self, database):
        """Initialize the retriever with a database connection.

        Parameters
        ----------
        database : SimulationDatabase
            Database instance to use for queries
        """
        self.db = database

    def __call__(self) -> T:
        """Execute the retriever's main functionality.

        Returns
        -------
        T
            The type returned by the specific retriever implementation
        """
        return self._execute()

    @abstractmethod
    def _execute(self) -> T:
        """Execute the retriever's main data retrieval and processing logic.

        This method must be implemented by all subclasses.

        Returns
        -------
        T
            The type returned by the specific retriever implementation

        Raises
        ------
        NotImplementedError
            If the subclass doesn't implement this method
        """
        raise NotImplementedError("Retriever must implement _execute method")

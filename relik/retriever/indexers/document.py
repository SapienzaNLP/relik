import csv
import json
from pathlib import Path
import pickle
import sys
from typing import Any, Dict, List, Union

from relik.common.log import get_logger
from relik.common.utils import JsonSerializable

csv.field_size_limit(sys.maxsize)

logger = get_logger(__name__)


class Document:
    def __init__(
        self,
        text: str,
        id: int | None = None,
        metadata: Dict | None = None,
        **kwargs,
    ):
        self.text = text
        # if id is not provided, we use the hash of the text
        self.id = id if id is not None else hash(text)
        # if metadata is not provided, we use an empty dictionary
        self.metadata = metadata or {}

    def __str__(self):
        return f"{self.id}:{self.text}"

    def __repr__(self):
        return  json.dumps(self.to_dict())

    def __eq__(self, other):
        if isinstance(other, Document):
            return self.id == other.id
        elif isinstance(other, int):
            return self.id == other
        elif isinstance(other, str):
            return self.text == other
        else:
            raise ValueError(
                f"Document must be compared with a Document, an int or a str, got `{type(other)}`"
            )

    def to_dict(self):
        return {"text": self.text, "id": self.id, "metadata": self.metadata}

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: Dict):
        return cls(**d)

    @classmethod
    def from_file(cls, file_path: Union[str, Path], **kwargs):
        with open(file_path, "r") as f:
            d = json.load(f)
        return cls.from_dict(d)

    def save(self, file_path: Union[str, Path], **kwargs):
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class DocumentStore:
    """
    A document store is a collection of documents.

    Args:
        documents (:obj:`List[Document]`):
            The documents to store.
    """

    def __init__(self, documents: List[Document] = None) -> None:
        if documents is None:
            documents = []
        # if self.ingore_case:
        #     documents = [doc.lower() for doc in documents]
        self._documents = documents
        # build an index for the documents
        self._documents_index = {doc.id: doc for doc in self._documents}
        # build a reverse index for the documents
        self._documents_reverse_index = {doc.text: doc for doc in self._documents}

    def __len__(self):
        return len(self._documents)

    def __getitem__(self, index):
        return self._documents[index]

    def __iter__(self):
        return iter(self._documents)

    def __contains__(self, item):
        if isinstance(item, int):
            return item in self._documents_index
        elif isinstance(item, str):
            return item in self._documents_reverse_index
        elif isinstance(item, Document):
            return item.id in self._documents_index
        # return item in self._documents_index

    def __str__(self):
        return f"DocumentStore with {len(self)} documents"

    def __repr__(self):
        return self.__str__()

    def get_document_from_id(self, id: int) -> Document | None:
        """
        Retrieve a document by its ID.

        Args:
            id (`int`):
                The ID of the document to retrieve.

        Returns:
            Optional[Document]: The document with the given ID, or None if it does not exist.
        """
        if id not in self._documents_index:
            logger.warning(f"Document with id `{id}` does not exist, skipping")
        return self._documents_index.get(id, None)

    def get_document_from_text(self, text: str) -> Document | None:
        """
        Retrieve the document by its text.

        Args:
            text (`str`):
                The text of the document to retrieve.

        Returns:
            Optional[Document]: The document with the given text, or None if it does not exist.
        """
        if text not in self._documents_reverse_index:
            logger.warning(f"Document with text `{text}` does not exist, skipping")
        return self._documents_reverse_index.get(text, None)

    def get_document_from_index(self, index: int) -> Document | None:
        """
        Retrieve the document by its index.

        Args:
            index (`int`):
                The index of the document to retrieve.

        Returns:
            Optional[Document]: The document with the given index, or None if it does not exist.
        """
        if index >= len(self._documents):
            logger.warning(f"Document with index `{index}` does not exist, skipping")
        return self._documents[index]

    def add_documents(
        self, documents: List[Document] | List[str] | List[Dict]
    ) -> List[Document]:
        """
        Add a list of documents to the document store.

        Args:
            documents (`List[Document]`):
                The documents to add.

        Returns:
            List[Document]: The documents just added.
        """
        return [
            (
                self.add_document(Document.from_dict(doc))
                if isinstance(doc, Dict)
                else self.add_document(doc)
            )
            for doc in documents
        ]

    def add_document(
        self,
        text: str | Document,
        id: int | None = None,
        metadata: Dict | None = None,
    ) -> Document:
        """
        Add a document to the document store.

        Args:
            text (`str`):
                The text of the document to add.
            id (`int`, optional, defaults to None):
                The ID of the document to add.
            metadata (`Dict`, optional, defaults to None):
                The metadata of the document to add.

        Returns:
            Document: The document just added.
        """
        if isinstance(text, str):
            # check if the document already exists
            if text in self:
                logger.warning(f"Document `{text}` already exists, skipping")
                return self._documents_reverse_index[text]
            if id is None:
                # get the len of the documents and add 1
                id = len(self._documents)  # + 1
            text = Document(text, id, metadata)

        if text in self:
            logger.warning(f"Document `{text}` already exists, skipping")
            return self._documents_index[text.id]

        self._documents.append(text)
        self._documents_index[text.id] = text
        self._documents_reverse_index[text.text] = text
        return text
        # if id in self._documents_index:
        #     logger.warning(f"Document with id `{id}` already exists, skipping")
        #     return self._documents_index[id]
        # if text_or_document in self._documents_reverse_index:
        #     logger.warning(f"Document with text `{text_or_document}` already exists, skipping")
        #     return self._documents_reverse_index[text_or_document]
        # self._documents.append(Document(text_or_document, id, metadata))
        # self._documents_index[id] = self._documents[-1]
        # self._documents_reverse_index[text_or_document] = self._documents[-1]
        # return self._documents_index[id]

    def delete_document(self, document: int | str | Document) -> bool:
        """
        Delete a document from the document store.

        Args:
            document (`int`, `str` or `Document`):
                The document to delete.

        Returns:
            bool: True if the document has been deleted, False otherwise.
        """
        if isinstance(document, int):
            return self.delete_by_id(document)
        elif isinstance(document, str):
            return self.delete_by_text(document)
        elif isinstance(document, Document):
            return self.delete_by_document(document)
        else:
            raise ValueError(
                f"Document must be an int, a str or a Document, got `{type(document)}`"
            )

    def delete_by_id(self, id: int) -> bool:
        """
        Delete a document by its ID.

        Args:
            id (`int`):
                The ID of the document to delete.

        Returns:
            bool: True if the document has been deleted, False otherwise.
        """
        if id not in self._documents_index:
            logger.warning(f"Document with id `{id}` does not exist, skipping")
            return False
        del self._documents_reverse_index[self._documents_index[id]]
        del self._documents_index[id]
        return True

    def delete_by_text(self, text: str) -> bool:
        """
        Delete a document by its text.

        Args:
            text (`str`):
                The text of the document to delete.

        Returns:
            bool: True if the document has been deleted, False otherwise.
        """
        if text not in self._documents_reverse_index:
            logger.warning(f"Document with text `{text}` does not exist, skipping")
            return False
        del self._documents_reverse_index[text]
        del self._documents_index[self._documents_index[text]]
        return True

    def delete_by_document(self, document: Document) -> bool:
        """
        Delete a document by its text.

        Args:
            document (:obj:`Document`):
                The document to delete.

        Returns:
            bool: True if the document has been deleted, False otherwise.
        """
        if document.id not in self._documents_index:
            logger.warning(f"Document {document} does not exist, skipping")
            return False
        del self._documents[self._documents.index(document)]
        del self._documents_index[document.id]
        del self._documents_reverse_index[self._documents_index[document.id]]

    def to_dict(self):
        return [doc.to_dict() for doc in self._documents]

    @classmethod
    def from_dict(cls, d):
        return cls([Document.from_dict(doc) for doc in d])

    @classmethod
    def from_file(cls, file_path: Union[str, Path], **kwargs):
        with open(file_path, "r") as f:
            # load a json lines file
            d = [Document.from_dict(json.loads(line)) for line in f]
        return cls(d)

    @classmethod
    def from_pickle(cls, file_path: Union[str, Path], **kwargs):
        with open(file_path, "rb") as handle:
            d = pickle.load(handle)
        return cls(d)

    @classmethod
    def from_tsv(
        cls,
        file_path: Union[str, Path],
        ingore_case: bool = False,
        delimiter: str = "\t",
        **kwargs,
    ):
        d = []
        # load a tsv/csv file and take the header into account
        # the header must be `id\ttext\t[list of metadata keys]`
        with open(file_path, "r", encoding="utf8") as f:
            csv_reader = csv.reader(f, delimiter=delimiter, **kwargs)
            header = next(csv_reader)
            id, text, *metadata_keys = header
            for i, row in enumerate(csv_reader):
                # check if id can be casted to int
                # if not, we add it to the metadata and use `i` as id
                try:
                    s_id = int(row[header.index(id)])
                    row_metadata_keys = metadata_keys
                except ValueError:
                    row_metadata_keys = [id] + metadata_keys
                    s_id = i

                d.append(
                    Document(
                        text=(
                            row[header.index(text)].strip().lower()
                            if ingore_case
                            else row[header.index(text)].strip()
                        ),
                        id=s_id,  # row[header.index(id)],
                        metadata={
                            key: row[header.index(key)] for key in row_metadata_keys
                        },
                    )
                )
        return cls(d)

    def save(self, file_path: Union[str, Path], **kwargs):
        with open(file_path, "w") as f:
            for doc in self._documents:
                # save as json lines
                f.write(json.dumps(doc.to_dict()) + "\n")

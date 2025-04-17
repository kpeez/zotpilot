"""Vector storage implementation using Milvus.

This module provides a unified interface for storing and retrieving
document chunks and their embeddings using Milvus as the backend.
"""

from pathlib import Path
from typing import Any

from pymilvus import (
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    model,
)

from ..utils.logging import get_component_logger
from .ingestion import (
    process_pdf_document,
)

DEFAULT_SIMILARITY_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.2


class VectorStore:
    """
    Handles interactions with a Milvus vector store collection for document chunks.

    Attributes:
        db_path (str): Path to the Milvus Lite database file.
        collection_name (str): Name of the collection managed by this instance.
        embedding_dim (int): Dimension of the vectors stored.
        client (MilvusClient): The initialized Milvus client.
        schema (CollectionSchema): The schema used for the collection.
    """

    DEFAULT_COLLECTION_NAME = "paperchat_docs"
    VECTOR_DB_DIR = str(Path.home() / ".paperchat" / "paperchat.db")

    def __init__(self):
        """
        Initializes the MilvusVectorStore, connects to Milvus, and ensures
        the specified collection and index exist.

        """
        self.db_path = self.VECTOR_DB_DIR
        self.collection_name = self.DEFAULT_COLLECTION_NAME
        # TODO: allow passing in different embedding functions
        self._embed_fn = model.DefaultEmbeddingFunction()
        self.embedding_dim = getattr(self._embed_fn, "dim", 768)
        self.logger = get_component_logger(f"MilvusVectorStore[{self.collection_name}]")
        try:
            self.client = MilvusClient(uri=self.db_path)
        except Exception as e:
            self.logger.exception(f"Failed to initialize MilvusClient: {e}")
            raise

        self.schema = self._define_schema()
        self.get_or_create_collection()
        self.build_index()

    def _define_schema(self) -> CollectionSchema:
        """Defines the Milvus collection schema."""
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(
                name="chunk_id",
                dtype=DataType.VARCHAR,
                max_length=64,
                description="Original chunk identifier",
            ),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=32768),
            FieldSchema(
                name="label", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_length=64
            ),
            FieldSchema(name="page_numbers", dtype=DataType.ARRAY, element_type=DataType.INT16),
            FieldSchema(name="headings", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(
                name="source",
                dtype=DataType.VARCHAR,
                max_length=512,
                description="Source document filename",
            ),
        ]

        return CollectionSchema(
            fields, description="Document Chunks Collection", enable_dynamic_field=True
        )

    def get_or_create_collection(self) -> None:
        """Creates the collection if it doesn't exist."""
        try:
            collection_exists = self.client.has_collection(self.collection_name)
            if not collection_exists:
                self.logger.info(f"Creating collection '{self.collection_name}'")
                self.client.create_collection(
                    collection_name=self.collection_name, schema=self.schema
                )

        except Exception as e:
            self.logger.exception(f"Failed to create collection: {e}")
            raise

    def build_index(self) -> None:
        """Creates a vector index on the embedding field."""
        try:
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type="IVF_FLAT",
                metric_type="COSINE",
                params={"nlist": 128},
            )

            self.client.create_index(
                collection_name=self.collection_name,
                index_params=index_params,
            )
        except Exception as e:
            self.logger.warning(f"Could not create index: {e}")

    def add_document(self, pdf_path: str | Path, max_tokens: int = 1024) -> bool:
        """
        Processes a PDF document, extracts chunks, generates embeddings,
        and inserts them into the Milvus collection. Automatically skips
        documents that have already been inserted.

        Args:
            pdf_path: Path to the PDF file.
            max_tokens: Maximum number of tokens per chunk

        Returns:
            True if the document was successfully processed and inserted, False otherwise
        """
        pdf_file = Path(pdf_path)
        self.logger.info(f"Processing: {pdf_file.name}")

        try:
            # check if document already exists in the collection
            existing = self.client.query(
                collection_name=self.collection_name,
                filter=f'source == "{pdf_file.stem}"',
                output_fields=["source"],
                limit=1,
            )

            if existing:
                self.logger.info(f"Document '{pdf_file.stem}' already exists. Skipping.")
                return True

            pdf_data = process_pdf_document(
                pdf_path, embed_fn=self._embed_fn, max_tokens_per_chunk=max_tokens
            )

            result = self.client.insert(collection_name=self.collection_name, data=pdf_data)
            insert_count = result.get("insert_count", 0)
            self.logger.info(f"Successfully inserted {insert_count} chunks")

            return True

        except Exception as e:
            self.logger.exception(f"Failed to insert data: {e}")

            return False

    def retrieve(
        self,
        query_text: str,
        top_k: int = DEFAULT_SIMILARITY_TOP_K,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        output_fields: list[str] | None = None,
        filter_expression: str | None = None,  # For metadata filtering later
    ) -> list[dict[str, Any]]:
        """
        Searches the collection for text chunks similar to the query text.

        Args:
            query_text: The text to search for.
            top_k: The maximum number of results to return.
            threshold: The minimum similarity score to return.
            output_fields: List of field names to include in the results.
                           Defaults to schema fields plus similarity/distance.
            filter_expression: Milvus filter expression string (e.g., "source == 'doc1.pdf'").

        Returns:
            Raw search results from Milvus
        """
        if output_fields is None:
            # default: all schema fields except embedding
            output_fields = [f.name for f in self.schema.fields if f.name != "embedding"]

        try:
            self.logger.debug(f"Loading collection '{self.collection_name}' for search.")
            self.client.load_collection(self.collection_name)
            query_embedding = self._embed_fn.encode_queries([query_text])[0]
        except Exception as e:
            self.logger.exception(f"Failed to generate query embedding or load collection: {e}")
            return []

        ########################################################################
        # radius and range_filter are used to filter results by distance
        # radius is boundary furthest from "ideal"
        # range_filter is boundary closest to "ideal"
        # for COSINE: radius <= similarity <= range_filter
        # for L2: radius >= similarity >= range_filter
        ########################################################################
        search_params = {
            "nprobe": 10,
            "metric_type": "COSINE",
            "radius": threshold,
            "range_filter": 1.0,
        }

        try:
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                search_params=search_params,
                limit=top_k,
                output_fields=output_fields,
                filter=filter_expression,
            )

        except Exception as e:
            self.logger.exception(f"Search failed: {e}")
            results = []

        return results

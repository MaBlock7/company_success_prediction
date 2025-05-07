from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType
from success_prediction.config import DATA_DIR


class DatabaseClient:
    def __init__(self, uri: str = DATA_DIR / 'database' / 'websites.db', collection_name: str = 'current_websites', **kwargs):
        self.milvus_client = MilvusClient(
            uri=str(uri),
            **kwargs
        )
        self.collection_name = collection_name
        self.default_schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="ehraid", dtype=DataType.INT64),
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=10),
            FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=5),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=64_000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=kwargs.get('dim', 768)),
        ])

    def setup_database(self, schema: CollectionSchema = None, replace: bool = False, **kwargs) -> None:
        """
        """
        if replace and self.milvus_client.has_collection(self.collection_name):
            self.milvus_client.drop_collection(self.collection_name)

        if not self.milvus_client.has_collection(self.collection_name):
            self.milvus_client.create_collection(
                collection_name=self.collection_name,
                schema=schema or self.default_schema,
                **kwargs
            )
        else:
            print(f"{self.collection_name} already exists!")

    def insert_data(self, data: list[dict]) -> None:
        self.milvus_client.insert(collection_name=self.collection_name, data=data)

    def search_by_query_embedding(
        self,
        query_embeddings: list[list[float]],
        top_k: int = 5,
        search_params: dict = {"metric_type": "COSINE", "params": {"nprobe": 10}},
        output_fields: list[str] = ["ehraid", "url", "text", "date", "language"],
        ehraid_filter: int | list[int] = None,
        **kwargs
    ) -> list[dict]:
        """
        Search Milvus with optional filter by ehraid (single int or list of ints).
        """
        # Construct the filter expression
        filter_expr = None
        if isinstance(ehraid_filter, int):
            filter_expr = f"ehraid == {ehraid_filter}"
        elif isinstance(ehraid_filter, list) and ehraid_filter:
            ids = ", ".join(str(i) for i in ehraid_filter)
            filter_expr = f"ehraid in [{ids}]"

        results = self.milvus_client.search(
            collection_name=self.collection_name,
            data=query_embeddings,
            anns_field="query_embedding",
            param=search_params,
            limit=top_k,
            output_fields=output_fields,
            filter=filter_expr,  # <- filter passed here
            kwargs=kwargs
        )

        # Flatten results
        hits = []
        for result in results:
            for hit in result:
                hits.append({
                    "score": hit.score,
                    **hit.entity
                })

        return hits

    def query_by_ehraid(
        self,
        filter: str,
        output_fields: list[str] = ["ehraid", "url", "date", "text", "language"]
    ) -> list[dict]:
        """
        Retrieve all rows for a specific company (ehraid) without vector search.
        """
        results = self.milvus_client.query(
            collection_name=self.collection_name,
            filter=filter,
            output_fields=output_fields
        )

        return results  # Already a list of dicts

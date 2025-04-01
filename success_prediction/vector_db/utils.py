from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType
from success_prediction.config import DATA_DIR


class DatabaseClient:
    def __init__(self, uri: str = DATA_DIR / 'database' / 'websites.db', collection_name: str = 'company_websites', **kwargs):
        self.milvus_client = MilvusClient(
            uri=uri,
            **kwargs
        )
        self.collection_name = collection_name
        self.default_schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="ehraid", dtype=DataType.INT64),
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=10),
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

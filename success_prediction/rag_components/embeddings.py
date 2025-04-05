import torch
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


class SimpleSplitter:

    def __init__(
        self,
        headers_to_split_on: list[tuple[str, str]] = [("#", "Header 1")],
        chunk_size=4096,
        chunk_overlap=128
    ):
        self.markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        self.recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_text(self, text: str) -> list[str]:
        md_header_splits = self.markdown_splitter.split_text(text)
        return self.recursive_splitter.split_documents(md_header_splits)


class EmbeddingCreator:
    def __init__(
        self,
        model_name: str,
        model_kwargs: dict = {},
        encode_kwargs: dict = {},
        detect_device: bool = True
    ):
        if detect_device:
            device = self._detect_device()
            model_kwargs.update({'device': device})
            print(f"[EmbeddingCreator] Using model on `{device}`.")

        self.embedder = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        self.splitter = SimpleSplitter()

    @staticmethod
    def _detect_device() -> str:
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    def chunk(self, text: str) -> list[str]:
        return self.splitter.split_text(text)

    def embed(self, documents: list[str]) -> list[list[float]]:
        return self.embedder.embed_documents(documents)

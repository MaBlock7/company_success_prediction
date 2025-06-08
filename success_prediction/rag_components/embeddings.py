from pathlib import Path
from unidecode import unidecode
from typing import Optional
import nltk
from nltk.corpus import stopwords
import numpy as np
import torch
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sentence_transformers import CrossEncoder


class SimpleSplitter:

    def __init__(self, headers_to_split_on=None, chunk_size=2560, chunk_overlap=256, **kwargs):
        """Initializes the SimpleSplitter with markdown and recursive character text splitters.

        Args:
            headers_to_split_on (list[tuple[str, str]]): Markdown headers to split on.
            chunk_size (int): Maximum number of characters in each chunk.
            chunk_overlap (int): Number of overlapping characters between chunks.
            **kwargs: Additional keyword arguments passed to the recursive splitter.
        """
        headers_to_split_on = headers_to_split_on or [("#", "Header 1"), ("##", "Header 2")]  # Default is to split on the first two header levels
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs
        )

    def merge_small_chunks(self, documents: list[Document], min_char: int = 100) -> list[Document]:
        """Merges chunks when a chunk is contains less than a minimum of characters to avoid
        ending up with small and uninformative chunks.

        Args:
            documents (list[Documents]): The chunks from the markdown splitter.

        Returns:
            list[Documents]: The merged chunks.
        """
        merged_docs = []
        buffer_content = ""
        buffer_metadata = {}

        for doc in documents:
            content = doc.page_content.strip()
            if len(content) < min_char:
                buffer_content += "\n" + content
                buffer_metadata.update(doc.metadata)
            else:
                if buffer_content:
                    # Merge buffer with current content
                    merged_content = buffer_content + "\n" + content
                    merged_docs.append(Document(page_content=merged_content.strip(), metadata=doc.metadata))
                    buffer_content = ""
                    buffer_metadata = {}
                else:
                    merged_docs.append(doc)

        # Handle leftover buffer at the end
        if buffer_content:
            merged_docs.append(Document(page_content=buffer_content.strip(), metadata=buffer_metadata))

        return merged_docs

    def split_text(self, text: str) -> list[str]:
        """Splits the input text into smaller chunks using markdown headers and recursive character splitting.

        Args:
            text (str): The input text to be split.

        Returns:
            list[str]: A list of split text chunks.
        """
        md_header_splits = self.markdown_splitter.split_text(text)
        md_combined = self.merge_small_chunks(md_header_splits)  # Avoid very smull chunks
        return self.recursive_splitter.split_documents(md_combined)


class EmbeddingHandler:
    def __init__(
        self,
        bi_encoder: str = 'intfloat/multilingual-e5-base',
        cross_encoder: str = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',
        bi_encoder_kwargs: dict = {},
        bi_encoder_encode_kwargs: dict = {},
        cross_encoder_kwargs: dict = {},
        splitter_kwargs: dict = {},
        detect_device: bool = True
    ):
        """Initializes the EmbeddingHandler with a bi-encoder, cross-encoder, and a text splitter.

        Args:
            bi_encoder (str): Model name for the bi-encoder used for embedding documents.
            cross_encoder (str): Model name for the cross-encoder used for relevance scoring.
            bi_encoder_kwargs (dict): Additional arguments for initializing the bi-encoder.
            bi_encoder_encode_kwargs (dict): Additional arguments for encoding with the bi-encoder.
            cross_encoder_kwargs (dict): Additional arguments for initializing the cross-encoder.
            splitter_kwargs (dict): Arguments for configuring the text splitter.
            detect_device (bool): Whether to automatically detect and use GPU/MPS/CPU.
        """
        if detect_device:
            self.device = self._detect_device()
            bi_encoder_kwargs.setdefault('device', self.device)
            cross_encoder_kwargs.setdefault('device', self.device)
            print(f"[EmbeddingHandler] Using model on `{self.device}`.")
        else:
            self.device = 'cpu'

        self.bi_encoder = HuggingFaceEmbeddings(
            model_name=bi_encoder,
            model_kwargs=bi_encoder_kwargs,
            encode_kwargs=bi_encoder_encode_kwargs
        )
        self.cross_encoder = CrossEncoder(
            model_name_or_path=cross_encoder,
            **cross_encoder_kwargs
        )
        self.splitter = SimpleSplitter(**splitter_kwargs)

    @staticmethod
    def _detect_device() -> str:
        """Detects the available device for computation (CUDA, MPS, or CPU).

        Returns:
            str: The name of the detected device.
        """
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    def chunk(self, text: str) -> list[str]:
        """Splits a given text into smaller chunks using the internal splitter.

        Args:
            text (str): The input text to be chunked.

        Returns:
            list[str]: A list of text chunks.
        """
        return self.splitter.split_text(text)

    def embed(self, documents: list[str], prefix: None | str = None) -> list[list[float]]:
        """Computes embeddings for a list of documents using the bi-encoder with optional an optional test prefix.

        Args:
            documents (list[str]): A list of documents to embed.
            prefix (str, optional): Optional prefix to prepend to each document.

        Returns:
            list[list[float]]: A list of embedding vectors.
        """
        if prefix:
            documents = [f'{prefix.strip()} {text}' for text in documents]
        return self.bi_encoder.embed_documents(documents)

    def calculate_relevancy_scores(self, sentence_pairs: list[tuple[str, str]]) -> torch.Tensor:
        """Computes relevance scores for a list of (query, paragraph) pairs using the cross-encoder.

        Args:
            sentence_pairs (list[tuple[str, str]]): List of (query, paragraph) pairs.

        Returns:
            torch.Tensor: A tensor of relevance scores.
        """
        scores = self.cross_encoder.predict(sentences=sentence_pairs)
        return torch.sigmoid(torch.tensor(scores))

    @staticmethod
    def waggregate_embeddings(embeddings: list[torch.Tensor], relevance_scores: list) -> torch.Tensor:
        """Aggregates a list of embeddings using softmax-weighted relevance scores.

        Args:
            embeddings (list[torch.Tensor]): A list of embedding tensors.
            relevance_scores (list): A list of relevance scores.

        Returns:
            torch.Tensor: The weighted average embedding.
        """
        embeddings = torch.stack(embeddings)
        weights = torch.softmax(torch.tensor(relevance_scores), dim=0).unsqueeze(1)
        return (weights * embeddings).sum(dim=0)

    @staticmethod
    def whitening_k(
        embeddings: torch.Tensor,
        k: Optional[int] = None,
        eps: float = 1e-9,
        reference: Optional[torch.Tensor] = None
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Performs Whitening-k (dimensionality-reduced whitening) on embeddings based on the algorithm proposed by su et al. (2021).

        Args:
            embeddings (torch.Tensor): Original embeddings of shape (N, D) with N = number of embeddings, D = embedding dimension.
            k (int, optional): Reserved dimensionality after whitening.
            eps (float): Small constant for numerical stability.

        Returns:
            torch.Tensor: Whitened and dimensionally reduced embeddings of shape (N, k).
        """
        # 1: Compute mean and covariance of the embeddings matrix row-wise
        mu = torch.mean(embeddings, dim=0, keepdim=True)  # -> outputs mean vector
        embeddings_centered = embeddings - mu  # (N, D)
        cov = embeddings_centered.T @ embeddings_centered / (embeddings_centered.shape[0] - 1)

        # 2: Compute SVD
        U, S, _ = torch.svd(cov)

        # 3: Compute Whitening matrix with dimensionality reduction to k dimensions
        if not k:
            k = embeddings.size(1)  # set k to original dimension, meaning no dimension reduction is applied
        W = (U[:, :k] @ torch.diag(1.0 / torch.sqrt(S[:k] + eps)))

        # 4: Apply transformation to each embedding
        embeddings_whitened = embeddings_centered @ W

        # 5 (optional): Apply same transform to reference vector
        ref_whitened = None
        if reference is not None:
            if reference.ndim == 1:
                reference = reference.unsqueeze(0)
            reference_centered = reference - mu
            ref_whitened = (reference_centered @ W).squeeze(0)

        return embeddings_whitened, ref_whitened


class Doc2VecHandler:
    """
    Implements the doc2vec similarity estimation as described in Guzman & Li (2023)
    """

    def __init__(self, ehraid2data: dict, language: str = 'german'):
        self.train_documents = None
        self.doc2vec_model = None
        self.ehraid2data = ehraid2data
        self.language = language
        self.stop_words = set(stopwords.words(language))

    def _prepare_train_documents(self):
        documents = []
        print("Preprocessing and tokenizing documents...")
        for ehraid, data in self.ehraid2data.items():
            date = data['date']
            text = ' '.join(data['text'])
            # Limit text length for computational reasons
            text = text[:400_000] if len(text) > 400_000 else text
            documents.append(
                (ehraid, date, self._preprocess_doc(text))
            )
        self.train_documents = documents

    def _preprocess_doc(self, doc: str) -> list[str]:
        tokens = nltk.tokenize.word_tokenize(doc.lower(), language=self.language)
        return [unidecode(w) for w in tokens if not w.lower() in self.stop_words]

    def train_doc2vec(self, vector_size: int = 300, window: int = 7, min_count: int = 3):
        if self.train_documents is None:
            self._prepare_train_documents()

        tagged_documents = [TaggedDocument(doc, [ehraid]) for ehraid, _, doc in self.train_documents]

        print("Training Doc2Vec model...")
        model = Doc2Vec(
            documents=tagged_documents,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4
        )
        print("Training complete.")
        self.doc2vec_model = model

    def save_doc2vec_model(self, folder_path: Path | str, filename: str = 'doc2vec') -> None:
        """
        Stores the trained doc2vec model in the given folder.
        The file will be saved as <folder_path>/<filename>.model
        """
        if not isinstance(folder_path, Path):
            folder_path = Path(folder_path)
        if self.doc2vec_model is None:
            raise ValueError("No trained doc2vec model to save.")
        folder_path.mkdir(parents=True, exist_ok=True)
        full_path = folder_path / f"{filename}_{self.language}.model"
        self.doc2vec_model.save(str(full_path))
        print(f"Doc2Vec model saved to {str(full_path)}")

    def create_normalized_vectors(self) -> dict[int, dict]:
        """
        Returns a dictionary mapping each ehraid to a dict with its normalized doc2vec vector and date.
        Example: {ehraid: {'vector': ..., 'date': ...}, ...}
        """
        vectors = self.doc2vec_model.dv.vectors
        ehraids = list(self.doc2vec_model.dv.index_to_key)
        mean_vec = np.mean(vectors, axis=0)
        normed_vectors = vectors - mean_vec
        return [
            (ehraid, self.ehraid2data[ehraid]['date'], normed_vectors[i])
            for i, ehraid in enumerate(ehraids)
        ]

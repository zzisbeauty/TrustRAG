import json
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional


class QdrantEngine:
    def __init__(self, collection_name: str, vector_size: int = 384, distance: Distance = Distance.COSINE):
        """
        Initialize the Qdrant vector store.

        :param collection_name: Name of the Qdrant collection.
        :param vector_size: Size of the vectors (default is 384 for all-MiniLM-L6-v2).
        :param distance: Distance metric for vector comparison (default is cosine similarity).
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance
        self.client = QdrantClient("http://localhost:6333")
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

        # Create collection if it doesn't exist
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=self.distance),
            )

    def upload_vectors(self, vectors_path: str, payload_path: str, batch_size: int = 256):
        """
        Upload vectors and payload to the Qdrant collection.

        :param vectors_path: Path to the .npy file containing the vectors.
        :param payload_path: Path to the .json file containing the payload.
        :param batch_size: Number of vectors to upload in a single batch.
        """
        # Load vectors from .npy file
        vectors = np.load(vectors_path)

        # Load payload from .json file
        with open(payload_path) as fd:
            payload = map(json.loads, fd)

        # Upload vectors and payload to Qdrant
        self.client.upload_collection(
            collection_name=self.collection_name,
            vectors=vectors,
            payload=payload,
            ids=None,  # Vector ids will be assigned automatically
            batch_size=batch_size,
        )

    def search(self, text: str, query_filter: Optional[Filter] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for the closest vectors in the collection based on the input text.

        :param text: The text query to search for.
        :param query_filter: Optional filter to apply to the search.
        :param limit: Number of closest results to return.
        :return: List of payloads from the closest vectors.
        """
        # Convert text query into vector
        vector = self.model.encode(text).tolist()

        # Search for closest vectors in the collection
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=query_filter,
            limit=limit,
        ).points

        # Extract payloads from search results
        payloads = [hit.payload for hit in search_result]
        return payloads

    def build_filter(self, conditions: List[Dict[str, Any]]) -> Filter:
        """
        Build a Qdrant filter from a list of conditions.

        :param conditions: A list of conditions, where each condition is a dictionary with:
                          - key: The field name to filter on.
                          - match: The value to match (can be a string, number, or other supported types).
        :return: A Qdrant Filter object.
        """
        filter_conditions = []
        for condition in conditions:
            key = condition.get("key")
            match_value = condition.get("match")
            if key and match_value is not None:
                filter_conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=match_value),
                    )
                )

        return Filter(must=filter_conditions)


# Example usage
if __name__ == "__main__":
    # Initialize the QdrantVectorStore
    vector_store = QdrantVectorStore(collection_name="startups")

    # Upload vectors and payload
    vector_store.upload_vectors(vectors_path="./startup_vectors.npy", payload_path="./startups_demo.json")

    # Build a filter for city and category
    conditions = [
        {"key": "city", "match": "Berlin"},
        {"key": "category", "match": "AI"},
    ]
    custom_filter = vector_store.build_filter(conditions)

    # Search for startups related to "AI" in Berlin
    results = vector_store.search(text="AI", query_filter=custom_filter, limit=5)
    for result in results:
        print(result)
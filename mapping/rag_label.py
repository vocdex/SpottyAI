import os
from typing import List, Dict, Optional
import logging
import base64
from io import BytesIO
from PIL import Image
import cv2
import argparse
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import json
from bosdyn.api.graph_nav import map_pb2
from label_waypoints import ClipAnnotationUpdater
from utils import get_map_paths

logger = logging.getLogger(__name__)

load_dotenv()
class MultimodalRAGAnnotator(ClipAnnotationUpdater):
    """
    Enhanced annotator that combines CLIP-based location classification with
    GPT-4V for detailed scene understanding and RAG capabilities.
    """
    def __init__(
        self,
        graph_file_path: str,
        snapshot_dir: str,
        vector_db_path: str = "vector_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        load_clip: bool = True,
        **kwargs
    ):
        super().__init__(graph_file_path, snapshot_dir, load_clip=load_clip, **kwargs)
        if not load_clip:
            logger.info("CLIP model not loaded")
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize vector store
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_db_path = vector_db_path
        self.vector_store = self._initialize_vector_store()

    def _initialize_vector_store(self) -> FAISS:
        """Initialize or load existing vector store."""
        import faiss
        
        if os.path.exists(self.vector_db_path):
            return FAISS.load_local(
                self.vector_db_path,
                self.embeddings,
                allow_dangerous_deserialization=True  # Add this parameter
            )
        
        # Create empty docstore
        empty_docs = [Document(page_content="", metadata={})]
        
        # Initialize FAISS with a sample document
        vectorstore = FAISS.from_documents(
            empty_docs,
            self.embeddings
        )
        
        
        return vectorstore
    

    def _encode_image_to_base64(self, cv_image: np.ndarray) -> str:
        """Convert OpenCV image to base64 string."""
        # Convert OpenCV image to PIL
        image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        
        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def analyze_scene_with_gpt4o(
        self,
        image: np.ndarray,
        prompt: str = "Describe this scene and list all visible objects. Focus on identifying objects that might be relevant for a robot to locate later."
    ) -> str:
        """
        Analyze scene using GPT-4o-mini.
        
        Args:
            image: OpenCV image
            prompt: Text prompt for GPT-4o-mini
        Returns:
            Scene description and object list
        """
        # Convert image to base64
        base64_image = self._encode_image_to_base64(image)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in GPT-4o-mini analysis: {e}")
            return "Error analyzing scene"

    def store_waypoint_data(
        self,
        waypoint_id: str,
        location: str,
        scene_descriptions: List[str],
        coordinates: Optional[Dict] = None
    ):
        """
        Store waypoint data in vector database.
        """
        # Make sure output directory exists
        os.makedirs(self.vector_db_path, exist_ok=True)
        
        # Process scene descriptions to extract object information
        combined_text = f"Location: {location}\n"
        combined_text += "Scene Descriptions:\n"
        for desc in scene_descriptions:
            combined_text += f"- {desc}\n"
        
        # Add coordinates if available
        if coordinates:
            combined_text += f"Coordinates: {json.dumps(coordinates)}\n"
        
        # Create document for vector store
        document = Document(
            page_content=combined_text,
            metadata={
                "waypoint_id": waypoint_id,
                "location": location,
                "coordinates": coordinates
            }
        )
        
        # Add to vector store
        self.vector_store.add_documents([document])
        
        # Save FAISS index
        faiss_index_path = os.path.join(self.vector_db_path, "index.faiss")
        self.vector_store.save_local(self.vector_db_path)
        
        # Log the save operation
        logger.info(f"Saved vector store to {self.vector_db_path}")
        logger.info(f"Number of documents in store: {len(self.vector_store.docstore._dict)}")

        # Save metadata separately for easier debugging
        metadata_path = os.path.join(self.vector_db_path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump({
                "total_documents": len(self.vector_store.docstore._dict),
                "last_added": {
                    "waypoint_id": waypoint_id,
                    "location": location,
                    "coordinates": coordinates
                }
            }, f, indent=2)

    def update_annotations_with_rag(self):
        """
        Enhanced annotation update using both CLIP and GPT-4V with RAG storage.
        """
        logger.info("Starting multimodal RAG annotation...")
        
        for waypoint in self.graph.waypoints:
            snapshot_path = os.path.join(self.snapshot_dir, waypoint.snapshot_id)
            
            if not os.path.exists(snapshot_path):
                logger.warning(f"Snapshot not found: {snapshot_path}")
                continue
            
            try:
                # Load snapshot
                snapshot = map_pb2.WaypointSnapshot()
                with open(snapshot_path, 'rb') as f:
                    snapshot.ParseFromString(f.read())
                
                # Get location using CLIP
                location = self.classify_waypoint_location(snapshot)
                
                # Process each camera image with GPT-4o-mini
                scene_descriptions = []
                for image in snapshot.images:
                    if image.source.name in self.DEPTH_CAMERA_SOURCES:
                        continue
                    
                    if image.source.name not in self.CAMERA_SOURCES:
                        continue

                    if image.source.name in self.FRONT_CAMERA_SOURCES:
                        opencv_image, _ = self.convert_image_from_snapshot(
                            image.shot.image,
                            image.source.name
                        )
                        
                        # Get detailed scene description using GPT-4o-mini
                        scene_desc = self.analyze_scene_with_gpt4o(opencv_image)
                        scene_descriptions.append(scene_desc)
                
                # Get coordinates from snapshot
                coordinates = {
                      "x" : waypoint.waypoint_tform_ko.position.x,
                      "y" : waypoint.waypoint_tform_ko.position.y,
                      "z" : waypoint.waypoint_tform_ko.position.z
                }
                
                # Store in vector database
                self.store_waypoint_data(
                    waypoint_id=waypoint.id,
                    location=location,
                    scene_descriptions=scene_descriptions,
                    coordinates=coordinates
                )
                
                # Update waypoint annotation
                waypoint.annotations.name = location
                logger.info(f"Processed waypoint {waypoint.id} as {location}")
            
            except Exception as e:
                logger.error(f"Error processing waypoint {waypoint.id}: {e}")
    
    def get_vector_store_info(self) -> Dict:
        """
        Get information about the current state of the vector store.
        """
        return {
            "total_documents": len(self.vector_store.docstore._dict),
            "vector_dimension": self.vector_store.index.d,
            "index_size": self.vector_store.index.ntotal,
            "store_path": self.vector_db_path
        }

    
    def query_location(
        self,
        query: str,
        k: int = 3,
        distance_threshold: float = 2.0  # L2 distance threshold (lower is better)
    ) -> List[Dict]:
        """
        Query the vector store for relevant waypoints.
        
        Args:
            query: Natural language query (e.g., "Where can I find a coffee mug?")
            k: Number of results to return
            distance_threshold: Maximum L2 distance to include in results
        Returns:
            List of relevant waypoints with metadata, sorted by distance
        """
        results = self.vector_store.similarity_search_with_score(
            query,
            k=k
        )
        
        # Convert results to list of dicts
        processed_results = [{
            'waypoint_id': doc.metadata['waypoint_id'],
            'location': doc.metadata['location'],
            'coordinates': doc.metadata['coordinates'],
            'description': doc.page_content,
            'distance': score  # Keep original L2 distance
        } for doc, score in results]
        
        # Filter by threshold and sort by distance (lower is better)
        filtered_results = [
            result for result in processed_results 
            if result['distance'] <= distance_threshold
        ]
        
        return sorted(filtered_results, key=lambda x: x['distance'])

    def print_query_results(self, results: List[Dict], max_results: int = 3):
        """
        Print query results in a readable format.
        
        Args:
            results: List of dictionaries containing query results
            max_results: Maximum number of results to print
        """
        if not results:
            print("No results found.")
            return
            
        print(f"\nFound {len(results)} relevant results:")
        
        # Print only up to max_results
        for i, result in enumerate(results[:max_results], 1):
            print(f"\n=== Result {i} (L2 Distance: {result['distance']:.3f}) ===")
            print(f"Waypoint ID: {result['waypoint_id']}")
            print("\nDescription:")
            desc_parts = result['description'].split('\n')
            for part in desc_parts:
                print(f"  {part}")
                
            print("="*50)


def main(args):
    graph_file_path, snapshot_dir, _ = get_map_paths(args.map_path)
    rag_annotator = MultimodalRAGAnnotator(
        graph_file_path=graph_file_path,
        snapshot_dir=snapshot_dir,
        vector_db_path="vector_db_chair",
        load_clip=False
    )

    # Update annotations and build RAG database
    # rag_annotator.update_annotations_with_rag()
    
    # Example query
    results = rag_annotator.query_location(
    "Where do you see kitchen setup?",
    k=5,                    # Get top 5 initial matches
    distance_threshold=3.0  # Only keep results with L2 distance < 2.0
    )
    rag_annotator.print_query_results(results, max_results=5)


# Example usage:
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Multimodal RAG Annotator")
    parser.add_argument("--map_path", type=str, help="Path to the map directory")
    parser.add_argument("--vector_db_path", type=str, default="vector_db", help="Path to the vector database")
    args = parser.parse_args()

    main(args)

    
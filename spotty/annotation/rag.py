#                                                                               
# Copyright (c) 2025, FAU, Shukrullo Nazirjonov
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#                                                                               
import base64
import json
import logging
import os
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional

import cv2
import numpy as np
from bosdyn.api.graph_nav import map_pb2
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field

from spotty.annotation.clip_manual import ClipAnnotationUpdater

load_dotenv()


# Separate class for RAG annotation using Pydantic
class WaypointAnnotation(BaseModel):
    visible_objects: List[str]
    hypothetical_questions: List[str] = Field(description="Hypothetical questions about the scene that could be asked")
    scene_description: str = Field(description="A brief description of the scene for robot")


class MultimodalRAGAnnotator(ClipAnnotationUpdater):
    """
    Enhanced annotator that combines CLIP-based location classification with
    GPT-4o-mini for detailed scene understanding and RAG capabilities.
    """

    def __init__(
        self,
        graph_file_path: str,
        logger: logging.Logger,
        snapshot_dir: str,
        vector_db_path: str = "vector_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        load_clip: bool = False,
        **kwargs,
    ):
        super().__init__(graph_file_path, logger, snapshot_dir, load_clip=load_clip, **kwargs)
        self.logger = logger

        if not load_clip:
            self.logger.info("CLIP model not loaded")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_db_path = vector_db_path
        self.vector_store = self._initialize_vector_store()

    def _initialize_vector_store(self) -> FAISS:
        """Initialize or load existing vector store."""

        if os.path.exists(self.vector_db_path):
            return FAISS.load_local(
                self.vector_db_path, self.embeddings, allow_dangerous_deserialization=True  # Add this parameter
            )

        empty_docs = [Document(page_content="", metadata={})]

        vectorstore = FAISS.from_documents(empty_docs, self.embeddings)
        return vectorstore

    def _encode_image_to_base64(self, cv_image: np.ndarray) -> str:
        """Convert OpenCV image to base64 string."""
        image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def analyze_scene_with_gpt4o(
        self,
        image: np.ndarray,
        prompt: str = """Your role is to find visible objects in the scene,
        describe the scene and ask hypothetical questions about the scene. Include 2 to 3 hypothetical questions.""",
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
            response = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                response_format=WaypointAnnotation,
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ],
                    },
                ],
                max_tokens=200,
            )

            return response.choices[0].message.parsed

        except Exception as e:
            self.logger.error(f"Error in GPT-4o-mini analysis: {e}")
            return "Error analyzing scene"

    def store_waypoint_data(
        self,
        waypoint_id: str,
        location: str,
        waypoint_annotations: Dict[str, WaypointAnnotation],  # Key is camera source name
        coordinates: Optional[Dict] = None,
    ):
        """
        Store waypoint data with separate left and right view annotations.

        Args:
            waypoint_id: Unique identifier for the waypoint
            location: Location name/label
            waypoint_annotations: Dict mapping camera source to WaypointAnnotation
            coordinates: Optional dictionary of x,y,z coordinates
        """
        os.makedirs(self.vector_db_path, exist_ok=True)

        combined_text = f"Location: {location}\n\n"

        # Process left and right views separately
        for camera_source, annotation in waypoint_annotations.items():
            view_name = "Left View" if "left" in camera_source.lower() else "Right View"
            combined_text += f"{view_name}:\n"
            combined_text += f"Scene Description: {annotation.scene_description}\n"
            combined_text += "Visible Objects:\n"
            for obj in annotation.visible_objects:
                combined_text += f"- {obj}\n"
            combined_text += "Questions About Scene:\n"
            for question in annotation.hypothetical_questions:
                combined_text += f"- {question}\n"
            combined_text += "\n"

        if coordinates:
            combined_text += f"Coordinates: {json.dumps(coordinates)}\n"

        # Create metadata with separate left/right objects
        metadata = {
            "waypoint_id": waypoint_id,
            "location": location,
            "coordinates": coordinates,
            "timestamp": datetime.now().isoformat(),
            "views": {},
        }

        # Store objects for each view separately in metadata
        for camera_source, annotation in waypoint_annotations.items():
            view_type = "left_view" if "left" in camera_source.lower() else "right_view"
            metadata["views"][view_type] = {
                "visible_objects": annotation.visible_objects,
                "camera_source": camera_source,
            }

        # Create document
        document = Document(page_content=combined_text, metadata=metadata)

        self.vector_store.add_documents([document])
        self.vector_store.save_local(self.vector_db_path)

        # Save detailed metadata
        metadata_path = os.path.join(self.vector_db_path, f"metadata_{waypoint_id}.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Saved vector store to {self.vector_db_path}")
        self.logger.info(f"Number of documents in store: {len(self.vector_store.docstore._dict)}")

    def update_annotations_with_rag(self):
        """
        Update annotations with separate left/right view processing.
        """
        self.logger.info("Starting multimodal RAG annotation...")
        print(f"Loaded graph with {len(self.graph.waypoints)} waypoints")

        for waypoint in self.graph.waypoints:
            print(f"Processing waypoint {waypoint.id}")
            snapshot_path = os.path.join(self.snapshot_dir, waypoint.snapshot_id)

            if not os.path.exists(snapshot_path):
                self.logger.warning(f"Snapshot not found: {snapshot_path}")
                continue

            try:
                snapshot = map_pb2.WaypointSnapshot()
                with open(snapshot_path, "rb") as f:
                    snapshot.ParseFromString(f.read())

                location = waypoint.annotations.name
                waypoint_annotations = {}

                for image in snapshot.images:
                    if image.source.name in self.DEPTH_CAMERA_SOURCES or image.source.name not in self.CAMERA_SOURCES:
                        continue

                    if image.source.name in self.FRONT_CAMERA_SOURCES:
                        opencv_image, _ = self.convert_image_from_snapshot(image.shot.image, image.source.name)

                        view_type = "left" if "left" in image.source.name.lower() else "right"
                        camera_specific_prompt = f"""Your role is to find visible objects in the scene from the front {view_type} camera view,
                        describe what you see from this {view_type} perspective, and ask hypothetical questions about the scene.
                        Include 2 to 3 hypothetical questions that are relevant to this specific view."""

                        annotation = self.analyze_scene_with_gpt4o(opencv_image, prompt=camera_specific_prompt)
                        if isinstance(annotation, WaypointAnnotation):
                            waypoint_annotations[image.source.name] = annotation

                coordinates = {
                    "x": waypoint.waypoint_tform_ko.position.x,
                    "y": waypoint.waypoint_tform_ko.position.y,
                    "z": waypoint.waypoint_tform_ko.position.z,
                }

                self.store_waypoint_data(
                    waypoint_id=waypoint.id,
                    location=location,
                    waypoint_annotations=waypoint_annotations,
                    coordinates=coordinates,
                )

                waypoint.annotations.name = location
                self.logger.info(f"Processed waypoint {waypoint.id} as {location}")

            except Exception as e:
                self.logger.error(f"Error processing waypoint {waypoint.id}: {e}")
                self.logger.exception(e)

    def get_vector_store_info(self) -> Dict:
        """
        Get information about the current state of the vector store.
        """
        return {
            "total_documents": len(self.vector_store.docstore._dict),
            "vector_dimension": self.vector_store.index.d,
            "index_size": self.vector_store.index.ntotal,
            "store_path": self.vector_db_path,
        }

    def query_location(
        self,
        query: str,
        filter_objects: Optional[List[str]] = None,
        k: int = 3,
        distance_threshold: float = 2.0,
        view_filter: Optional[str] = None,  # 'left', 'right', or None for both
    ) -> List[Dict]:
        """
        Query function with separate left/right view filtering.

        Args:
            query: Natural language query
            filter_objects: Optional list of objects to filter results by
            k: Number of results to return
            distance_threshold: Maximum L2 distance for results
            view_filter: Optional filter for specific view ('left', 'right', or None)
        """
        results = self.vector_store.similarity_search_with_score(query, k=k)

        processed_results = []
        for doc, score in results:
            # Get views data from metadata
            views_data = doc.metadata.get("views", {})

            # Apply view filter if specified
            if view_filter:
                view_key = f"{view_filter}_view"
                if view_key not in views_data:
                    continue
                views_data = {view_key: views_data[view_key]}

            # Check object filter across specified views
            if filter_objects:
                objects_in_views = set()
                for view_data in views_data.values():
                    objects_in_views.update(view_data.get("visible_objects", []))
                if not any(obj in objects_in_views for obj in filter_objects):
                    continue

            result = {
                "waypoint_id": doc.metadata["waypoint_id"],
                "location": doc.metadata["location"],
                "coordinates": doc.metadata["coordinates"],
                "description": doc.page_content,
                "distance": score,
                "timestamp": doc.metadata.get("timestamp"),
                "views": views_data,
            }
            processed_results.append(result)

        filtered_results = [result for result in processed_results if result["distance"] <= distance_threshold]

        return sorted(filtered_results, key=lambda x: x["distance"])

    def get_waypoint_annotations(self, waypoint_id: str) -> Optional[Dict]:
        """
        Retrieve annotations for a specific waypoint by exact ID match.

        Args:
            waypoint_id: The exact waypoint ID to search for

        Returns:
            Optional[Dict]: Dictionary containing all annotations for the waypoint,
                        or None if waypoint not found
        """
        try:
            # Search through all documents in the vector store
            all_docs = self.vector_store.docstore._dict

            for doc_id, doc in all_docs.items():
                # Check if this document's metadata contains our waypoint_id
                if doc.metadata.get("waypoint_id") == waypoint_id:
                    return {
                        "waypoint_id": waypoint_id,
                        "location": doc.metadata.get("location"),
                        "coordinates": doc.metadata.get("coordinates"),
                        "timestamp": doc.metadata.get("timestamp"),
                        "views": doc.metadata.get("views", {}),
                        "full_description": doc.page_content,
                    }

            self.logger.warning(f"No annotations found for waypoint ID: {waypoint_id}")
            return None

        except Exception as e:
            self.logger.error(f"Error retrieving annotations for waypoint {waypoint_id}: {e}")
            self.logger.exception(e)
            return None

    def print_query_results(self, results: List[Dict], max_results: int = 3):
        """
        Print results with complete information for each view.
        """
        if not results:
            print("No results found.")
            return

        print(f"\nFound {len(results)} relevant locations:")

        for i, result in enumerate(results[:max_results], 1):
            print(f"\n=== Result {i} (L2 Distance: {result['distance']:.3f}) ===")
            print(f"Waypoint ID: {result['waypoint_id']}")
            # print(f"Location: {result['location']}")

            # Parse and print the content by view
            content_sections = result["description"].split("\n\n")
            for section in content_sections:
                if section.strip():  # Only print non-empty sections
                    print(f"\n{section.strip()}")

            if result.get("coordinates"):
                print("\nCoordinates:")
                for key, value in result["coordinates"].items():
                    print(f"  {key}: {value:.2f}")

            print("=" * 50)

#!/usr/bin/env python3
#                                                                               
# Copyright (c) 2025, FAU, Shukrullo Nazirjonov
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#                                                                               
"""
Synchronize RAG metadata and rebuild vector store with GraphNav waypoint labels.
Usage: python sync_rag_data.py --map-path /path/to/map --rag-path /path/to/rag_db
"""

import argparse
import json
import logging
import os
import shutil

from bosdyn.api.graph_nav import map_pb2
from langchain.schema import Document

from spotty.annotation import MultimodalRAGAnnotator


def setup_logger():
    """Set up logger for the sync process."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def load_metadata_files(rag_path: str) -> dict:
    """Load all existing metadata files."""
    metadata_dict = {}
    metadata_files = [f for f in os.listdir(rag_path) if f.startswith("metadata_") and f.endswith(".json")]

    for metadata_file in metadata_files:
        try:
            with open(os.path.join(rag_path, metadata_file), "r") as f:
                metadata = json.load(f)
                waypoint_id = metadata.get("waypoint_id")
                if waypoint_id:
                    metadata_dict[waypoint_id] = metadata
        except Exception as e:
            print(f"Error loading {metadata_file}: {e}")

    return metadata_dict


def rebuild_vector_store(rag_path: str, metadata_dict: dict, rag: MultimodalRAGAnnotator):
    """Remove old vector store files and rebuild from metadata."""
    # Remove existing vector store files
    index_file = os.path.join(rag_path, "index.faiss")
    pkl_file = os.path.join(rag_path, "index.pkl")

    if os.path.exists(index_file):
        os.remove(index_file)
    if os.path.exists(pkl_file):
        os.remove(pkl_file)

    print("\nRebuilding vector store...")
    documents = []

    for waypoint_id, metadata in metadata_dict.items():
        # Create the combined text content as before
        combined_text = f"Location: {metadata['location']}\n\n"

        # Process views
        for view_type, view_data in metadata["views"].items():
            view_name = "Left View" if "left" in view_type.lower() else "Right View"
            combined_text += f"{view_name}:\n"
            if "visible_objects" in view_data:
                combined_text += "Visible Objects:\n"
                for obj in view_data["visible_objects"]:
                    combined_text += f"- {obj}\n"
            combined_text += "\n"

        if "coordinates" in metadata:
            combined_text += f"Coordinates: {json.dumps(metadata['coordinates'])}\n"

        # Create document with full metadata
        document = Document(page_content=combined_text, metadata=metadata)
        documents.append(document)

    # Create new vector store from documents
    rag.vector_store = rag.vector_store.from_documents(documents, rag.embeddings)

    # Save the new vector store
    rag.vector_store.save_local(rag_path)
    print(f"Vector store rebuilt with {len(documents)} documents")


# Add force_rebuild parameter to sync_rag_data function
def sync_rag_data(map_path: str, rag_path: str, logger, force_rebuild: bool = False):
    """Synchronize RAG data with graph labels and optionally force rebuild vector store."""
    try:
        # Initialize RAG annotator
        rag = MultimodalRAGAnnotator(
            graph_file_path=os.path.join(map_path, "graph"),
            logger=logger,
            snapshot_dir=os.path.join(map_path, "waypoint_snapshots"),
            vector_db_path=rag_path,
            load_clip=False,
        )

        # Load graph
        graph = map_pb2.Graph()
        with open(os.path.join(map_path, "graph"), "rb") as f:
            graph.ParseFromString(f.read())

        # Load existing metadata
        existing_metadata = load_metadata_files(rag_path)

        print(f"\nProcessing {len(graph.waypoints)} waypoints...")
        updated_count = 0
        unchanged_count = 0
        error_count = 0

        # Process each waypoint
        for waypoint in graph.waypoints:
            try:
                waypoint_id = waypoint.id
                new_location = waypoint.annotations.name

                if waypoint_id not in existing_metadata:
                    print(f"Warning: No existing metadata for waypoint {waypoint_id}")
                    error_count += 1
                    continue

                metadata = existing_metadata[waypoint_id]
                current_location = metadata.get("location", "")

                if current_location == new_location:
                    unchanged_count += 1
                    continue

                # Update the metadata with new location
                metadata["location"] = new_location
                existing_metadata[waypoint_id] = metadata

                # Update the metadata file
                metadata_file = os.path.join(rag_path, f"metadata_{waypoint_id}.json")
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)

                print(f"Updated {waypoint_id}: {current_location} -> {new_location}")
                updated_count += 1

            except Exception as e:
                print(f"Error processing waypoint {waypoint_id}: {e}")
                error_count += 1

        if updated_count > 0 or force_rebuild:
            # Rebuild vector store with updated metadata
            print("\nRebuilding vector store..." + (" (forced)" if force_rebuild and updated_count == 0 else ""))
            rebuild_vector_store(rag_path, existing_metadata, rag)
        else:
            print("\nNo updates detected and force rebuild not requested. Vector store remains unchanged.")

        # Print summary
        print("\nSync Summary:")
        print(f"  Updated: {updated_count} waypoints")
        print(f"  Unchanged: {unchanged_count} waypoints")
        print(f"  Errors: {error_count} waypoints")
        print(f"  Total: {updated_count + unchanged_count + error_count} waypoints processed")

        # Verify vector store update
        store_info = rag.get_vector_store_info()
        print("\nVector Store Status:")
        print(f"  Total documents: {store_info['total_documents']}")
        print(f"  Vector dimension: {store_info['vector_dimension']}")
        print(f"  Index size: {store_info['index_size']}")
        print(f"\nVector store files location: {rag_path}")

    except Exception as e:
        print(f"Error during synchronization: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Sync RAG metadata and vector store with GraphNav labels")
    parser.add_argument("--map-path", type=str, required=True, help="Path to GraphNav map directory")
    parser.add_argument("--rag-path", type=str, required=True, help="Path to RAG database directory")
    parser.add_argument(
        "--force-rebuild", action="store_true", help="Force rebuild of vector store even if no updates are detected"
    )
    args = parser.parse_args()

    logger = setup_logger()
    sync_rag_data(args.map_path, args.rag_path, logger, force_rebuild=args.force_rebuild)


if __name__ == "__main__":
    main()

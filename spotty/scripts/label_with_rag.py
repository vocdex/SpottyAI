import logging
import argparse
from dotenv import load_dotenv
from spotty.annotation.rag import MultimodalRAGAnnotator
from spotty.utils.common import get_map_paths

logger = logging.getLogger(__name__)

load_dotenv()


def main(args):
    graph_file_path, snapshot_dir, _ = get_map_paths(args.map_path)
    rag_annotator = MultimodalRAGAnnotator(
        graph_file_path=graph_file_path,
        logger=logger,
        snapshot_dir=snapshot_dir,
        vector_db_path=args.vector_db_path,
        load_clip=False
    )

    # Update annotations and build RAG database
    rag_annotator.update_annotations_with_rag()
    
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
    parser.add_argument("--map-path", type=str, help="Path to the map directory")
    parser.add_argument("--vector-db-path", type=str, default="vector_db", help="Path to the vector database")
    args = parser.parse_args()

    main(args)
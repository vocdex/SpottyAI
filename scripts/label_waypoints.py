"""Given a recorded GraphNav graph, this script updates the waypoint annotations using two methods:
1. Manual Annotation: Update annotations using a dictionary of custom labels.
2. CLIP Annotation: Automatically label waypoints using CLIP model based on waypoint snapshots.
"""
import argparse
import logging

from spotty.annotation.clip_manual import ClipAnnotationUpdater, ManualAnnotationUpdater
from spotty.utils.common_utils import get_map_paths, read_manual_labels

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main(args):
    """
    Enhanced main function with more robust error handling.
    """
    try:
        # Path resolution
        graph_file_path, snapshot_dir, output_graph_path = get_map_paths(args.map_path)

        # Choose annotation method
        if args.label_method == "manual":
            if not args.label_file:
                raise ValueError("Label file is required for manual annotation")

            manual_updater = ManualAnnotationUpdater(graph_file_path, logger=logger)
            manual_labels = read_manual_labels(args.label_file)

            manual_updater.print_waypoint_annotations()
            manual_updater.update_annotations(manual_labels)
            manual_updater.save_updated_graph(
                output_dir=output_graph_path,
            )

        elif args.label_method == "clip":
            updater = ClipAnnotationUpdater(
                graph_file_path,
                logger,
                snapshot_dir,
                text_prompts=args.prompts,
                clip_model=args.clip_model,
                use_multiprocessing=args.parallel,
                confidence_threshold=args.confidence_threshold,
                neighbor_threshold=args.neighbor_threshold,
            )
            updater.update_annotations()
            updater.save_updated_graph(output_dir=output_graph_path)

    except Exception as e:
        logger.error(f"Annotation process failed: {e}")
        raise


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Advanced GraphNav Waypoint Labeling")
    arg_parser.add_argument("--map-path", type=str, required=True, help="Path to the GraphNav map directory")
    arg_parser.add_argument(
        "--label-method", type=str, required=True, choices=["manual", "clip"], help="Labeling method"
    )
    arg_parser.add_argument("--label-file", type=str, help="Path to the custom label file for manual annotation")

    # New arguments for enhanced flexibility
    arg_parser.add_argument("--prompts", type=str, nargs="+", help="Custom location prompts for CLIP")
    arg_parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32", help="CLIP model to use")
    arg_parser.add_argument("--parallel", action="store_true", help="Enable multiprocessing")
    arg_parser.add_argument("--confidence-threshold", type=float, default=0.6, help="Confidence threshold for CLIP")
    arg_parser.add_argument("--neighbor-threshold", type=float, default=0.6, help="Neighbor threshold for CLIP")

    args = arg_parser.parse_args()
    # Time the main function
    import time

    start_time = time.perf_counter()
    main(args)
    end_time = time.perf_counter()
    print(f"Annotation process completed in {end_time - start_time:.2f} seconds.")

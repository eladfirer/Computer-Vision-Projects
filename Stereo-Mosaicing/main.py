"""CLI entry point for video mosaicing pipeline."""

import argparse
import logging
import sys
from pathlib import Path

from src.config import MosaicConfig
from src.pipeline import VideoMosaicPipeline
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def run_single_config(config_path: str) -> bool:
    """Run a single configuration file.
    
    Args:
        config_path: Path to configuration JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        config = MosaicConfig.from_json(config_path)
        logger.info(f"Configuration loaded from {config_path}")
        
        pipeline = VideoMosaicPipeline(config)
        output_path = pipeline.run()
        
        logger.info(f"Success! Output video: {output_path}")
        return True
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return False
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return False
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return False


def run_all_configs() -> int:
    """Run all config*.json files in the current directory.
    
    Returns:
        Exit code (0 if all successful, 1 otherwise)
    """
    config_files = sorted(Path('.').glob('config*.json'))
    
    if not config_files:
        logger.error("No config*.json files found in current directory")
        return 1
    
    logger.info(f"Found {len(config_files)} config files")
    logger.info("Will generate video mosaics for each config")
    
    total_runs = len(config_files)
    current_run = 0
    failed_runs: list[str] = []
    
    for config_file in config_files:
        current_run += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"[{current_run}/{total_runs}] Processing {config_file.name}")
        logger.info(f"{'='*60}")
        
        success = run_single_config(str(config_file))
        if not success:
            failed_runs.append(config_file.name)
            logger.error(f"Failed to process {config_file.name}")
        else:
            logger.info(f"Successfully completed {config_file.name}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total configs: {total_runs}")
    logger.info(f"Successful: {total_runs - len(failed_runs)}")
    logger.info(f"Failed: {len(failed_runs)}")
    
    if failed_runs:
        logger.warning("\nFailed configs:")
        for failed in failed_runs:
            logger.warning(f"  - {failed}")
        return 1
    else:
        logger.info("\nAll configs completed successfully!")
        return 0


def main() -> int:
    """Main entry point for CLI.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(
        description="Generate video mosaics from input videos using optical flow tracking"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON configuration file (required unless --all-configs is used)"
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Run all config*.json files in the current directory"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    # Handle --all-configs option
    if args.all_configs:
        return run_all_configs()
    
    # Single config mode
    if not args.config:
        parser.error("Either --config or --all-configs must be specified")
    
    return 0 if run_single_config(args.config) else 1


if __name__ == "__main__":
    sys.exit(main())


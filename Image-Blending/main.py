#!/usr/bin/env python3
"""Command-line interface for Image Blending and Hybrid Image creation.

This script provides a clean CLI for:
- Interactive image blending with Laplacian pyramids
- Hybrid image creation
- FFT comparison analysis
"""

import argparse
import sys
import cv2

from src.pipelines.blending import image_blending, blend_with_provided_mask
from src.pipelines.hybrid import hybrid_images
from src.utils.visualization import save_fft_comparison_with_colorbar


def main() -> None:
    """Main entry point for the image blending CLI."""
    parser = argparse.ArgumentParser(
        description="Advanced Image Blending and Hybrid Image Creation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive image blending (draw mask manually)
  python main.py imageblending img1.jpg img2.jpg

  # Image blending with provided mask
  python main.py imageblending img1.jpg img2.jpg --mask mask.png

  # Create hybrid image
  python main.py hybridimages cat.jpg dog.jpg

  # Compare FFT spectra of two images
  python main.py compare-fft img1.jpg img2.jpg
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Processing mode')
    
    # Image blending subcommand
    blend_parser = subparsers.add_parser(
        'imageblending',
        help='Blend two images using Laplacian pyramid blending'
    )
    blend_parser.add_argument(
        'image1',
        type=str,
        help='Path to first image'
    )
    blend_parser.add_argument(
        'image2',
        type=str,
        help='Path to second image'
    )
    blend_parser.add_argument(
        '--mask',
        type=str,
        default=None,
        help='Path to binary mask (optional, if not provided will draw interactively)'
    )
    blend_parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode: generate pyramid visualizations and FFT analysis'
    )
    
    # Hybrid images subcommand
    hybrid_parser = subparsers.add_parser(
        'hybridimages',
        help='Create a hybrid image from two input images'
    )
    hybrid_parser.add_argument(
        'image1',
        type=str,
        help='Path to low-frequency image (seen from far)'
    )
    hybrid_parser.add_argument(
        'image2',
        type=str,
        help='Path to high-frequency image (seen from close)'
    )
    
    # FFT comparison subcommand
    fft_parser = subparsers.add_parser(
        'compare-fft',
        help='Compare FFT spectra of two images'
    )
    fft_parser.add_argument(
        'image1',
        type=str,
        help='Path to first image'
    )
    fft_parser.add_argument(
        'image2',
        type=str,
        help='Path to second image'
    )
    fft_parser.add_argument(
        '--output',
        type=str,
        default='fft_comparison.png',
        help='Output filename for FFT comparison (default: fft_comparison.png)'
    )
    
    args = parser.parse_args()
    
    if args.mode is None:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.mode == 'imageblending':
            debug = getattr(args, 'debug', False)
            if args.mask:
                blend_with_provided_mask(args.image1, args.image2, args.mask, debug=debug)
            else:
                image_blending(args.image1, args.image2, debug=debug)
        
        elif args.mode == 'hybridimages':
            hybrid_images(args.image1, args.image2)
        
        elif args.mode == 'compare-fft':
            img_a = cv2.imread(args.image1)
            img_b = cv2.imread(args.image2)
            
            if img_a is None:
                raise FileNotFoundError(f"Could not load: {args.image1}")
            if img_b is None:
                raise FileNotFoundError(f"Could not load: {args.image2}")
            
            save_fft_comparison_with_colorbar(img_a, img_b, args.output)
            print(f"FFT comparison saved to: {args.output}")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


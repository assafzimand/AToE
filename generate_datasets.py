"""Script to generate datasets for all problems."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.io import load_config
from utils.dataset_gen import generate_and_save_datasets


def main():
    """Generate datasets with visualizations."""
    print("=" * 60)
    print("Dataset Generation")
    print("=" * 60)
    
    # Load config
    config = load_config()
    
    # Generate datasets for the current problem
    print(f"\nGenerating datasets for problem: {config['problem']}")
    generate_and_save_datasets(config)
    
    print("\n" + "=" * 60)
    print("✓ Dataset generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()


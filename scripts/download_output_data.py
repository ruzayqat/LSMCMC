#!/usr/bin/env python3
"""
Download pre-computed output data from Zenodo for figure reproduction.

This script downloads the pre-computed DA output (~1.8 GB) required for
reproducing figures in the LSMCMC.ipynb notebook without re-running the
full experiments (which take hours/days).

Usage:
    python3 download_output_data.py

The data will be extracted to the current directory, creating:
    - output_lsmcmc_ldata_V1/
    - output_lsmcmc_ldata_V2/
    - output_letkf_ldata_K25/
    - output_lsmcmc_nldata_twin_V1/
    - output_lsmcmc_nldata_twin_V2/
    - output_nlgamma_twin_V1/
    - output_nlgamma_twin_V2/
    - linear_gaussian/*.npz  (updated)
    - ... etc.

Author: Hamza Ruzayqat
"""

import os
import sys
import zipfile
import hashlib
from pathlib import Path

# =============================================================================
# Configuration — Update ZENODO_RECORD_ID after uploading to Zenodo
# =============================================================================
ZENODO_RECORD_ID = "18889551"  # Data DOI: 10.5281/zenodo.18889551
ZENODO_FILENAME = "lsmcmc_output_data.zip"
EXPECTED_SHA256 = None  # TODO: Set after upload for integrity verification

# Zenodo download URL pattern
ZENODO_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/{ZENODO_FILENAME}?download=1"

# Alternative: Direct URL if hosted elsewhere
DIRECT_URL = None  # Set this for non-Zenodo hosting


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> None:
    """Download a file with progress indicator."""
    import urllib.request
    
    print(f"Downloading from: {url}")
    print(f"Destination: {dest}")
    
    try:
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            
            with open(dest, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        pct = downloaded * 100 / total_size
                        mb_down = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        print(f"\r  Progress: {mb_down:.1f}/{mb_total:.1f} MB ({pct:.1f}%)", end="", flush=True)
            
            print()  # Newline after progress
            
    except Exception as e:
        print(f"\nError downloading: {e}")
        raise


def verify_checksum(filepath: Path, expected: str) -> bool:
    """Verify SHA256 checksum of downloaded file."""
    if expected is None:
        print("  Skipping checksum verification (no expected hash provided)")
        return True
    
    print("  Verifying checksum...")
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    
    actual = sha256.hexdigest()
    if actual == expected:
        print("  Checksum OK")
        return True
    else:
        print(f"  Checksum MISMATCH!")
        print(f"    Expected: {expected}")
        print(f"    Got:      {actual}")
        return False


def extract_archive(archive: Path, dest_dir: Path) -> None:
    """Extract zip archive to destination directory."""
    print(f"Extracting to: {dest_dir}")
    
    with zipfile.ZipFile(archive, 'r') as zf:
        members = zf.namelist()
        total = len(members)
        
        for i, member in enumerate(members, 1):
            zf.extract(member, dest_dir)
            if i % 100 == 0 or i == total:
                print(f"\r  Extracted: {i}/{total} files", end="", flush=True)
        
        print()  # Newline after progress


def check_existing_data(base_dir: Path) -> bool:
    """Check if output data already exists."""
    required_dirs = [
        "output_lsmcmc_ldata_V1",
        "output_lsmcmc_ldata_V2", 
        "output_letkf_ldata_K25",
    ]
    
    for d in required_dirs:
        if (base_dir / d).exists():
            return True
    
    # Also check for linear_gaussian npz files
    lg_dir = base_dir / "linear_gaussian"
    if lg_dir.exists():
        npz_files = list(lg_dir.glob("linear_gaussian_lsmcmc*.npz"))
        if npz_files:
            return True
    
    return False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download pre-computed LSMCMC output data")
    parser.add_argument("--noninteractive", action="store_true",
                        help="Run without prompts (for Binder/CI)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if data exists")
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.resolve()
    archive_path = base_dir / ZENODO_FILENAME
    
    print("=" * 60)
    print("LSMCMC Output Data Downloader")
    print("=" * 60)
    
    # Check if data already exists
    if check_existing_data(base_dir) and not args.force:
        print("\nPre-computed output data already exists.")
        if args.noninteractive:
            print("Skipping download (use --force to override).")
            return
        response = input("Re-download and overwrite? [y/N]: ").strip().lower()
        if response != 'y':
            print("Skipping download.")
            return
    
    # Check configuration
    if ZENODO_RECORD_ID == "XXXXXXX":
        print("\n" + "!" * 60)
        print("ERROR: Zenodo record ID not configured!")
        print()
        print("This script requires pre-computed output data to be uploaded")
        print("to Zenodo first. Please:")
        print("  1. Create a zip archive of output_*/ folders and linear_gaussian/*.npz")
        print("  2. Upload to Zenodo and get the record ID")
        print("  3. Update ZENODO_RECORD_ID in this script")
        print()
        print("For now, you can run the experiments manually:")
        print("  - Linear Gaussian: cd linear_gaussian && python3 linear_forward_generate_data.py")
        print("  - MLSWE: python3 run_mlswe_lsmcmc_ldata_V1.py (takes hours)")
        print("!" * 60)
        sys.exit(1)
    
    # Determine URL
    url = DIRECT_URL if DIRECT_URL else ZENODO_URL
    
    # Download
    print(f"\nDownloading pre-computed output data (~1.8 GB)...")
    try:
        download_file(url, archive_path)
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)
    
    # Verify checksum
    if not verify_checksum(archive_path, EXPECTED_SHA256):
        print("Checksum verification failed. The download may be corrupted.")
        sys.exit(1)
    
    # Extract
    print("\nExtracting archive...")
    try:
        extract_archive(archive_path, base_dir)
    except Exception as e:
        print(f"Extraction failed: {e}")
        sys.exit(1)
    
    # Clean up archive
    print(f"\nCleaning up: removing {archive_path.name}")
    archive_path.unlink()
    
    print("\n" + "=" * 60)
    print("SUCCESS! Output data is ready.")
    print("You can now run the notebook to reproduce figures.")
    print("=" * 60)


if __name__ == "__main__":
    main()

import sys
from pathlib import Path
import shutil
import os

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))



TARGET_FILES = [
    'HI-Small_Trans.csv',
    'HI-Small_accounts.csv',
    'HI-Small_Patterns.txt'
]

RAW_DATA_DIR = project_root / 'data' / 'raw'

def download_with_kaggle_api():
    """
    Download using kagglehub library.
    Requires kaggle.json in ~/.kaggle/
    """
    try:
        import kagglehub
        
        print(" Downloading dataset from Kaggle...")
        print("   This downloads ALL files first, we'll keep only what we need\n")
        
        path = kagglehub.dataset_download("ealtman2019/ibm-transactions-for-anti-money-laundering-aml")
        
        print(f"\n Downloaded to: {path}")
        print(" Moving target files to data/raw/...\n")
        
        move_target_files(Path(path))
        
    except ImportError:
        print(" kagglehub not installed")
        print("   Run: pip install kagglehub")
        print("   Or use manual download (see instructions below)")
    except Exception as e:
        print(f" Download failed: {e}")
        print("\n Use manual download instead (see instructions below)")

def move_target_files(download_path):
    """
    Move only the files we need to data/raw/
    Deletes the rest to save disk space.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    for filename in TARGET_FILES:
        # Search for the file (might be in a subfolder)
        matches = list(download_path.rglob(filename))
        
        if matches:
            src = matches[0]
            dest = RAW_DATA_DIR / filename
            
            print(f"   Moving {filename}...")
            shutil.move(str(src), str(dest))
            
            size_mb = dest.stat().st_size / (1024**2)
            print(f"      {filename} → data/raw/ ({size_mb:.1f} MB)")
        else:
            print(f"    {filename} not found in download")
    
    print(f"\n Files ready in: {RAW_DATA_DIR}")

def print_manual_instructions():
    """Print manual download instructions"""

    
    print("""
     Steps:
   1. Go to: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
   2. Sign in (free account)
   3. Click "Download" (top right)
   4. Extract the .zip file
   5. Move ONLY these files to data\\raw\\:
   
       HI-Small_Trans.csv      (650 MB) - Main transactions
       HI-Small_accounts.csv   (47 MB)  - Account details
       HI-Small_Patterns.txt   (tiny)   - Patterns reference
    

    """)
    
    print(f" Target folder: {RAW_DATA_DIR}")
    print("=" * 70)

def verify_data():
    """Check if all required files are in place"""
    
    print("\n Checking data/raw/ contents:")
    print("-" * 50)
    
    all_good = True
    
    for filename in TARGET_FILES:
        filepath = RAW_DATA_DIR / filename
        
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024**2)
            print(f"   {filename} ({size_mb:.1f} MB)")
        else:
            print(f"   {filename} (missing)")
            all_good = False
    
    if all_good:
        print("\n All files present! Ready for ingestion.")
    else:
        print("\n  Some files are missing. Follow manual download instructions.")
        print_manual_instructions()
    
    return all_good

if __name__ == "__main__":
    # First check if we already have the files
    if verify_data():
        print("\n Data already downloaded.")
        sys.exit(0)
    
    # Try API download first, fall back to instructions
    print("\n Attempting Kaggle API download...\n")
    
    try:
        download_with_kaggle_api()
    except Exception:
        print_manual_instructions()
    
    # Final check
    verify_data()
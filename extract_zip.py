import os
import zipfile
import shutil
from pathlib import Path

def safe_extract_zip(zip_path, extract_path):
    """Safely extract ZIP file contents while preserving directory structure."""
    
    def safe_path(path):
        """Clean path components and create safe path."""
        # Convert to Path object and resolve any ../ or ./ components
        path = Path(path)
        # Remove any parent directory references and hidden paths
        clean_parts = [part for part in path.parts if part not in ('..', '.', '~') and not part.startswith('.')]
        if not clean_parts:
            return None
        # Handle case when first part is the project folder name
        if clean_parts[0] == 'Roboto.SAI-Roboto_SAI-Beta':
            clean_parts = clean_parts[1:]
        return os.path.join(*clean_parts) if clean_parts else None

    # Create extraction directory
    os.makedirs(extract_path, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get list of files in zip
        file_list = zip_ref.namelist()
        
        # Extract each file safely
        for file in file_list:
            try:
                # Clean and validate path
                safe_file_path = safe_path(file)
                if not safe_file_path:
                    print(f"Skipping invalid path: {file}")
                    continue
                
                # Create full extraction path
                full_path = os.path.join(extract_path, safe_file_path)
                full_dir = os.path.dirname(full_path)
                
                # Create parent directories
                os.makedirs(full_dir, exist_ok=True)
                
                # Skip if it's a directory
                if file.endswith('/'):
                    continue
                
                # Extract file
                source = zip_ref.open(file)
                with open(full_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                
                print(f"Successfully extracted: {full_path}")
                
            except Exception as e:
                print(f"Error extracting {file}: {str(e)}")

if __name__ == "__main__":
    zip_path = r"C:\Users\ytkro\OneDrive\Documents\Roboto.SAI-Roboto_SAI-Beta.zip"
    extract_path = r"C:\Users\ytkro\OneDrive\Desktop\Roboto.SAI-Beta\Roboto.SAI"
    safe_extract_zip(zip_path, extract_path)
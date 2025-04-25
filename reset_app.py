import os
import shutil
import torch

def clear_cache_and_reset():
    print("Clearing all caches and resetting app...")
    
    # Clear model cache
    cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'torch', 'sentence_transformers')
    if os.path.exists(cache_dir):
        print(f"Removing model cache: {cache_dir}")
        try:
            shutil.rmtree(cache_dir)
            print("Model cache cleared!")
        except Exception as e:
            print(f"Error clearing model cache: {e}")
    
    # Clear database
    db_path = os.path.join('instance', 'educhat.db')
    if os.path.exists(db_path):
        print(f"Removing database: {db_path}")
        try:
            os.remove(db_path)
            print("Database removed!")
        except Exception as e:
            print(f"Error removing database: {e}")
    
    # Clear vector database
    vector_db_dir = "vector_db_data"
    if os.path.exists(vector_db_dir):
        print(f"Clearing vector database directory: {vector_db_dir}")
        try:
            shutil.rmtree(vector_db_dir)
            os.makedirs(vector_db_dir)
            print("Vector database directory cleared!")
        except Exception as e:
            print(f"Error clearing vector database: {e}")
    
    # Clear uploads
    uploads_dir = "uploads"
    if os.path.exists(uploads_dir):
        print(f"Clearing uploads directory: {uploads_dir}")
        try:
            for f in os.listdir(uploads_dir):
                file_path = os.path.join(uploads_dir, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print("Uploads directory cleared!")
        except Exception as e:
            print(f"Error clearing uploads: {e}")
    
    # Clear torch cache
    print("Clearing PyTorch cache...")
    try:
        torch.cuda.empty_cache()
        print("PyTorch cache cleared!")
    except:
        print("No CUDA cache to clear")
    
    # Clear __pycache__ folders
    for root, dirs, files in os.walk("."):
        for dir in dirs:
            if dir == "__pycache__":
                pycache_path = os.path.join(root, dir)
                print(f"Removing: {pycache_path}")
                try:
                    shutil.rmtree(pycache_path)
                except Exception as e:
                    print(f"Error removing {pycache_path}: {e}")
    
    print("All caches cleared! Run 'flask run' to start a fresh instance.")

if __name__ == "__main__":
    clear_cache_and_reset()
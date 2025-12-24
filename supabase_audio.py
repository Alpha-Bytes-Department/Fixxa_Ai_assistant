import os
from dotenv import load_dotenv
from supabase import create_client, Client
from typing import List, Optional
import requests

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)


def list_audio_files(bucket_name: str, folder_path: str = "") -> List[dict]:
    """
    List all audio files in a Supabase storage bucket.
    
    Args:
        bucket_name: Name of the storage bucket
        folder_path: Optional folder path within the bucket (e.g., "audio/recordings")
    
    Returns:
        List of file objects with metadata
    """
    try:
        files = supabase.storage.from_(bucket_name).list(folder_path)
        return files
    except Exception as e:
        print(f"Error listing files: {e}")
        return []


def get_audio_file_url(bucket_name: str, file_path: str, expires_in: int = 3600) -> Optional[str]:
    """
    Get a signed URL for an audio file (for private buckets).
    
    Args:
        bucket_name: Name of the storage bucket
        file_path: Full path to the file in the bucket (e.g., "audio/file.wav")
        expires_in: URL expiration time in seconds (default: 1 hour)
    
    Returns:
        Signed URL string or None if error
    """
    try:
        url = supabase.storage.from_(bucket_name).create_signed_url(file_path, expires_in)
        return url['signedURL']
    except Exception as e:
        print(f"Error getting signed URL: {e}")
        return None


def get_public_audio_url(bucket_name: str, file_path: str) -> str:
    """
    Get a public URL for an audio file (for public buckets).
    
    Args:
        bucket_name: Name of the storage bucket
        file_path: Full path to the file in the bucket
    
    Returns:
        Public URL string
    """
    try:
        url = supabase.storage.from_(bucket_name).get_public_url(file_path)
        return url
    except Exception as e:
        print(f"Error getting public URL: {e}")
        return None


def download_audio_file(bucket_name: str, file_path: str, local_path: str) -> bool:
    """
    Download an audio file from Supabase storage to local disk.
    
    Args:
        bucket_name: Name of the storage bucket
        file_path: Full path to the file in the bucket
        local_path: Local path where file should be saved
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Download file as bytes
        file_data = supabase.storage.from_(bucket_name).download(file_path)
        
        # Write to local file
        with open(local_path, 'wb') as f:
            f.write(file_data)
        
        print(f"File downloaded successfully to {local_path}")
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False


def upload_audio_file(bucket_name: str, file_path: str, local_file_path: str) -> bool:
    """
    Upload an audio file to Supabase storage.
    
    Args:
        bucket_name: Name of the storage bucket
        file_path: Destination path in the bucket (e.g., "audio/recording.wav")
        local_file_path: Path to the local file to upload
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(local_file_path, 'rb') as f:
            supabase.storage.from_(bucket_name).upload(file_path, f)
        
        print(f"File uploaded successfully to {file_path}")
        return True
    except Exception as e:
        print(f"Error uploading file: {e}")
        return False


def delete_audio_file(bucket_name: str, file_path: str) -> bool:
    """
    Delete an audio file from Supabase storage.
    
    Args:
        bucket_name: Name of the storage bucket
        file_path: Full path to the file in the bucket
    
    Returns:
        True if successful, False otherwise
    """
    try:
        supabase.storage.from_(bucket_name).remove([file_path])
        print(f"File {file_path} deleted successfully")
        return True
    except Exception as e:
        print(f"Error deleting file: {e}")
        return False


def get_most_recent_audio(status: str, bucket_name: str = "audio_storage") -> Optional[dict]:
    """
    Fetch the most recent audio file based on status.
    
    Args:
        status: Either "quote" or "invoice"
        bucket_name: Name of the storage bucket (default: "audio_storage")
    
    Returns:
        Dictionary with file info including 'name', 'path', and 'url', or None if error
    """
    try:
        # Determine folder based on status
        if status.lower() == "quote":
            folder_path = "quote_audio"
        elif status.lower() == "invoice":
            folder_path = "quote_recordings"
        else:
            print(f"Invalid status: {status}. Must be 'quote' or 'invoice'")
            return None
        
        # List all files in the folder
        files = supabase.storage.from_(bucket_name).list(folder_path)
        
        if not files:
            print(f"No files found in {folder_path}")
            return None
        
        # Sort by created_at or updated_at to get the most recent
        # Filter out folders (they don't have 'created_at' or have id as null)
        audio_files = [f for f in files if f.get('id') is not None]
        
        if not audio_files:
            print(f"No audio files found in {folder_path}")
            return None
        
        # Sort by created_at timestamp (most recent first)
        most_recent = sorted(audio_files, key=lambda x: x.get('created_at', ''), reverse=True)[0]
        
        # Build full path
        file_path = f"{folder_path}/{most_recent['name']}"
        
        # Get signed URL for the file
        signed_url = get_audio_file_url(bucket_name, file_path)
        
        result = {
            'name': most_recent['name'],
            'path': file_path,
            'url': signed_url,
            'created_at': most_recent.get('created_at'),
            'size': most_recent.get('metadata', {}).get('size', 0)
        }
        
        print(f"Most recent {status} audio: {most_recent['name']}")
        return result
        
    except Exception as e:
        print(f"Error fetching most recent audio: {e}")
        return None


# Example usage
if __name__ == "__main__":
    # Replace with your bucket name
    BUCKET_NAME = "audio_storage"
    
    # List all files in the bucket
    print("Listing files...")
    files = list_audio_files(BUCKET_NAME)
    for file in files:
        print(f"- {file['name']}")
    
    # Get most recent quote audio
    print("\n--- Testing get_most_recent_audio ---")
    quote_audio = get_most_recent_audio("quote")
    if quote_audio:
        print(f"Quote Audio: {quote_audio}")
    
    # Get most recent invoice audio
    invoice_audio = get_most_recent_audio("invoice")
    if invoice_audio:
        print(f"Invoice Audio: {invoice_audio}")
    
    # Download a specific file
    # download_audio_file(BUCKET_NAME, "recordings/audio.wav", "downloaded_audio.wav")
    
    # Get a signed URL for private file
    # url = get_audio_file_url(BUCKET_NAME, "recordings/audio.wav")
    # print(f"Signed URL: {url}")
    
    # Upload a file
    # upload_audio_file(BUCKET_NAME, "recordings/new_audio.wav", "recorded_audio.wav")

import os
import requests
import hashlib
from urllib.parse import urljoin
import cv2
import numpy as np
import trafilatura
from bs4 import BeautifulSoup

def download_image(url, save_path):
    """Download and validate image from URL"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return False

        # Convert to numpy array for validation
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return False
            
        # Validate image size
        if image.shape[0] < 100 or image.shape[1] < 100:
            return False

        # Save image
        cv2.imwrite(save_path, image)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def get_image_urls(query, num_pages=10):
    """Get image URLs from web search"""
    base_urls = [
        "https://commons.wikimedia.org/wiki/Category:Aloe_vera",
        "https://pixabay.com/images/search/aloe%20vera/",
        "https://www.pexels.com/search/aloe%20vera/"
    ]
    
    image_urls = set()
    
    for base_url in base_urls:
        try:
            downloaded = trafilatura.fetch_url(base_url)
            if downloaded:
                soup = BeautifulSoup(downloaded, 'html.parser')
                for img in soup.find_all('img'):
                    src = img.get('src')
                    if src:
                        if not src.startswith(('http://', 'https://')):
                            src = urljoin(base_url, src)
                        if src.lower().endswith(('.jpg', '.jpeg', '.png')):
                            image_urls.add(src)
        except Exception as e:
            print(f"Error scraping {base_url}: {str(e)}")
            continue
            
    return list(image_urls)

def download_dataset(save_dir='data/training/Aloe_Vera', num_images=1000):
    """Download and save dataset images"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Get image URLs
    image_urls = get_image_urls('aloe vera plant')
    
    # Download images
    count = 0
    for url in image_urls:
        if count >= num_images:
            break
            
        # Generate filename from URL
        filename = hashlib.md5(url.encode()).hexdigest() + '.jpg'
        save_path = os.path.join(save_dir, filename)
        
        # Skip if file exists
        if os.path.exists(save_path):
            continue
            
        # Download and save image
        if download_image(url, save_path):
            count += 1
            print(f"Downloaded {count} images")
    
    return count

if __name__ == "__main__":
    num_downloaded = download_dataset()
    print(f"Successfully downloaded {num_downloaded} images")

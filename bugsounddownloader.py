import requests
from bs4 import BeautifulSoup
import os
import urllib.parse
import urllib.request
import re

def download_wav_files(url, base_dir="..\\..\\bugsounds"):
    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Fetch the webpage
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return

    # Parse HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all links
    links = soup.find_all('a')
    
    for link in links:
        # Get link text and href
        link_text = link.get_text().strip()
        href = link.get('href')
        
        if not link_text or not href:
            continue
            
        # Create valid folder name from link text
        folder_name = re.sub(r'[<>:"/\\|?*]', '_', link_text)
        
        # Skip if folder name would be empty
        if not folder_name:
            continue
            
        # Resolve relative URLs
        full_url = urllib.parse.urljoin(url, href)
        
        try:
            # Fetch the linked page
            linked_response = requests.get(full_url, headers=headers)
            linked_response.raise_for_status()
            linked_soup = BeautifulSoup(linked_response.text, 'html.parser')
            
            # Find all links that end in .wav
            wav_links = []
            for tag in linked_soup.find_all(['a', 'source']):
                wav_url = tag.get('href') or tag.get('src')
                if wav_url and wav_url.lower().endswith('.wav'):
                    wav_links.append(urllib.parse.urljoin(full_url, wav_url))
            
            if wav_links:
                # Create folder if WAV files found
                folder_path = os.path.join(base_dir, folder_name)
                os.makedirs(folder_path, exist_ok=True)
                print(f"\nDownloading WAV files for: {link_text}")
                
                # Download each WAV file
                for wav_url in wav_links:
                    try:
                        filename = os.path.join(folder_path, os.path.basename(wav_url))
                        print(f"Downloading: {wav_url}")
                        wav_response = requests.get(wav_url, headers=headers, stream=True)
                        wav_response.raise_for_status()
                        with open(filename, 'wb') as f:
                            for chunk in wav_response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        print(f"Saved to: {filename}")
                    except Exception as e:
                        print(f"Error downloading {wav_url}: {e}")
                        
        except requests.RequestException as e:
            print(f"Error fetching link {full_url}: {e}")
            continue

if __name__ == "__main__":
    url = input("Enter the webpage URL: ")
    download_wav_files(url)

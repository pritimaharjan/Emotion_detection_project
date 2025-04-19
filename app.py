import requests
import os

# Replace 'your_access_key' with your actual Unsplash API access key
ACCESS_KEY = 'hi8feXEwEKz9A7wKqlwqOrOexZdqSdb1642dYhZUVBQ'
UNSPLASH_URL = 'https://api.unsplash.com/search/photos'

def download_images(query, download_folder='happy_images_3', per_page=30, total_images=180):
    headers = {
        'Authorization': f'Client-ID {ACCESS_KEY}'
    }
    
    # Create download folder if it doesn't exist
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    
    downloaded_images = 0
    page = 1
    
    while downloaded_images < total_images:
        params = {
            'query': query,
            'page': page,
            'per_page': per_page
        }
        
        response = requests.get(UNSPLASH_URL, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            images = data['results']
            
            if not images:
                print("No more images found.")
                break
            
            for image in images:
                image_url = image['urls']['full']
                image_id = image['id']
                image_filename = f"{download_folder}/{image_id}.jpg"
                
                # Download the image
                image_response = requests.get(image_url)
                if image_response.status_code == 200:
                    with open(image_filename, 'wb') as file:
                        file.write(image_response.content)
                    downloaded_images += 1
                    print(f"Downloaded image {downloaded_images}/{total_images}: {image_filename}")
                else:
                    print(f"Failed to download image {image_id}")
                
                if downloaded_images >= total_images:
                    break
            
            page += 1
        else:
            print(f"Failed to fetch images. Status code: {response.status_code}")
            break

if __name__ == "__main__":
    topic = input("Enter the topic for images: ")
    download_images(topic)
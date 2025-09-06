import requests

data = {
    "relative_path": "ugc/video1.mp4",
    "callback_url": "https://webhook.site/b21acdec-da33-4442-994a-d36df6ea77a9"
}

response = requests.post("http://localhost:8000/generate-thumbnail/", json=data)
print(response.json())

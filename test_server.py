
import requests
import json
import time

URL = "http://localhost:10000"

def test_health():
    print(f"Testing health check at {URL}/health...")
    try:
        res = requests.get(f"{URL}/health")
        print(f"Status: {res.status_code}")
        print(f"Response: {json.dumps(res.json(), indent=2)}")
    except Exception as e:
        print(f"Health check failed: {e}")

def test_search():
    query = "What is the recommended treatment for diabetes?"
    # NOTE: You must replace 'your_collection_name' with a real collection in your Qdrant
    collection = "medical_guidelines" 
    
    print(f"\nTesting search at {URL}/search...")
    print(f"Query: '{query}', Collection: '{collection}'")
    
    payload = {
        "query": query,
        "collection_name": collection,
        "limit": 3
    }
    
    try:
        res = requests.post(f"{URL}/search", json=payload)
        if res.status_code == 200:
            data = res.json()
            results = data.get("results", [])
            print(f"Success! Found {len(results)} results.")
            if results:
                print("First result snippet:")
                print(json.dumps(results[0], indent=2))
        else:
            print(f"Search failed with {res.status_code}: {res.text}")
    except Exception as e:
        print(f"Search request failed: {e}")

if __name__ == "__main__":
    print("--- Nomic RAG Server Tester ---")
    print("Ensure the server is running on localhost:10000")
    test_health()
    
    # Prompt user before running search test as it requires keys
    val = input("\nDo you want to run the search test? (Requires valid keys in server .env) [y/N]: ")
    if val.lower() == 'y':
        test_search()

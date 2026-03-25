import httpx

url = "http://lamb-server:8000/api/instruments"
try:
    with httpx.Client() as client:
        # Try some common discovery endpoints
        for endpoint in ["/api/instruments", "/instruments", "/list"]:
            resp = client.get(f"http://lamb-server:8000{endpoint}")
            print(f"Endpoint {endpoint}: {resp.status_code}")
            if resp.status_code == 200:
                print(resp.text)
except Exception as e:
    print(f"Discovery failed: {e}")

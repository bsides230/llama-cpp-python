import json

def test_json(data):
    try:
        json.dumps(data)
    except Exception as e:
        print(f"Failed: {e}")

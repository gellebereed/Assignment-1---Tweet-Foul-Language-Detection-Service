import os, sys
from fastapi.testclient import TestClient

# Make the service package importable for tests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from service.app import app

client = TestClient(app)

def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json().get('status') == 'ok'

def test_predict_happy(monkeypatch):
    from service import app as appmod
    class DummyPipeline:
        def predict_proba(self, X):
            import numpy as np
            return np.array([[0.8, 0.2]])  # mostly proper
    appmod.artifact = {'pipeline': DummyPipeline(), 'threshold': 0.5, 'label_map': {0:'proper',1:'foul'}}
    r = client.post('/predict', json={'text': 'I love this app'})
    assert r.status_code == 200
    assert r.json()['label'] == 0

def test_predict_foul(monkeypatch):
    from service import app as appmod
    class DummyPipeline:
        def predict_proba(self, X):
            import numpy as np
            return np.array([[0.1, 0.9]])  # mostly foul
    appmod.artifact = {'pipeline': DummyPipeline(), 'threshold': 0.5, 'label_map': {0:'proper',1:'foul'}}
    r = client.post('/predict', json={'text': 'You are stupid!!!'})
    assert r.status_code == 200
    assert r.json()['label'] == 1

def test_invalid_input():
    r = client.post('/predict', json={'text': ''})
    assert r.status_code == 422

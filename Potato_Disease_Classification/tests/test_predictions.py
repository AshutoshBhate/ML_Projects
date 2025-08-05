import pytest
from unittest.mock import Mock
from PIL import Image
import io


def create_fake_jpeg_image_bytes(size=(256, 256)):
    img = Image.new('RGB', size, 'red')
    byte_io = io.BytesIO()
    img.save(byte_io, format='JPEG')
    byte_io.seek(0)
    return byte_io.getvalue()


def test_predict_unauthenticated(client):
    
    fake_image_data = b"this is not a real image"
    files = {'file': ('test.jpg', fake_image_data, 'image/jpeg')}
    
    res = client.post("/predictions/", files=files)
    assert res.status_code == 401 

def test_predict_success(authorized_client, mocker):

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "predictions": [[0.1, 0.9, 0.0]] # Simulate a prediction for "Late Blight"
    }
    mocker.patch("api.routers.prediction.requests.post", return_value=mock_response)
    
    fake_image_data = create_fake_jpeg_image_bytes()
    files = {'file': ('leaf.jpg', fake_image_data, 'image/jpeg')}
    
    res = authorized_client.post("/predictions/", files=files)
    
    assert res.status_code == 200
    prediction = res.json()
    assert prediction['predicted_class'] == "Late Blight"
    assert prediction['confidence'] == pytest.approx(0.9)
    assert prediction['filename'] == 'leaf.jpg'

def test_get_prediction_history(authorized_client, session, test_user, mocker):

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"predictions": [[0.9, 0.1, 0.0]]}
    mocker.patch("api.routers.prediction.requests.post", return_value=mock_response)
    
    fake_image_data = create_fake_jpeg_image_bytes()
    files = {'file': ('history_test.jpg', fake_image_data, 'image/jpeg')}
    res_post = authorized_client.post("/predictions/", files=files)
    
    assert res_post.status_code == 200
    
    res_get = authorized_client.get("/predictions/")
    assert res_get.status_code == 200
    history = res_get.json()
    
    assert len(history) == 1
    assert history[0]['predicted_class'] == "Early Blight"
    assert history[0]['owner']['id'] == test_user['id']
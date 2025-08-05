#Workflow trigger for Gihub Actions!!!

from api import schemas

def test_create_user(client):
    res = client.post("/users/", json={"email": "hello@world.com", "password": "password123"})
    assert res.status_code == 201
    
    new_user = schemas.UserCreateResponse(**res.json())
    assert new_user.email == "hello@world.com"

def test_login_user(client, test_user):
    res = client.post("/login", data={"username": test_user['email'], "password": test_user['password']})
    
    assert res.status_code == 200
    token = schemas.Token(**res.json())
    assert token.token_type == 'bearer'
    assert token.access_token is not None

def test_login_invalid_credentials(client, test_user):
    res = client.post("/login", data={"username": test_user['email'], "password": "wrongpassword"})
    assert res.status_code == 403
    assert res.json()['detail'] == "Invalid Credentials"
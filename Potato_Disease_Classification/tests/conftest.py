import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from api.database import Base, get_db
from api.main_Routers import app
from api.config import settings

TEST_DATABASE_URL = (
    f"postgresql://{settings.database_username}:{settings.database_password}@"
    f"{settings.database_hostname}:{settings.database_port}/{settings.database_name}_test"
)

engine = create_engine(TEST_DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

@pytest.fixture(scope="function")
def session():

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

@pytest.fixture(scope="function")
def client(session):
    def override_get_db():
        try:
            yield session
        finally:
            session.close()
    
    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)

@pytest.fixture
def test_user_data():
    return {"email": "testuser@example.com", "password": "password123"}

@pytest.fixture
def test_user(client, test_user_data):
    res = client.post("/users/", json=test_user_data)
    assert res.status_code == 201
    new_user = res.json()
    new_user['password'] = test_user_data['password']
    return new_user

@pytest.fixture
def authorized_client(client, test_user):
    login_data = {"username": test_user['email'], "password": test_user['password']}
    res = client.post("/login", data=login_data)
    assert res.status_code == 200
    token = res.json()['access_token']
    
    client.headers = {
        **client.headers,
        "Authorization": f"Bearer {token}"
    }
    return client
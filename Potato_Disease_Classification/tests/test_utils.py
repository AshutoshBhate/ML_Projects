from api import utils

def test_password_hashing():
    password = "mysecretpassword"
    hashed_password = utils.hash(password)
    
    assert password != hashed_password
    assert utils.verify(password, hashed_password) == True
    assert utils.verify("wrongpassword", hashed_password) == False
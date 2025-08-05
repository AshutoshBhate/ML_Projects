# from pydantic_settings import BaseSettings

# class Settings(BaseSettings):
#     # Database Configuration
#     DATABASE_URL: str

#     # TensorFlow Serving Configuration
#     TF_SERVING_URL: str

#     # Authentication Configuration (for later steps)
#     SECRET_KEY: str
#     ALGORITHM: str
#     ACCESS_TOKEN_EXPIRE_MINUTES: int

#     class Config:
#         env_file = ".env"

# # Create a single, importable instance of the settings
# settings = Settings()

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
   
    database_hostname: str
    database_port: str
    database_password: str
    database_name: str
    database_username: str
    

    tf_serving_url: str

    secret_key: str
    algorithm: str
    access_token_expire_minutes: int
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
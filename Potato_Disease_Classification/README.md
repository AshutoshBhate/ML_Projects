# Potato Disease Classification Full-Stack Project

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-red.svg)
![Docker](https://img.shields.io/badge/Docker-24.0-blue.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13-blue.svg)
![CI/CD](https://github.com/AshutoshBhate/ML_Projects/actions/workflows/ci.yml/badge.svg)

This project is a complete full-stack application designed to classify potato diseases from leaf images. It features a deep learning model for classification, a high-performance RESTful API to serve the model, a secure user authentication system with JWT, and an interactive web interface for users. The entire system is containerized with Docker for easy deployment and includes a CI pipeline with GitHub Actions for automated testing.

---

## Table of Contents
- [Project Architecture](#project-architecture)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Machine Learning Model](#machine-learning-model)
- [Testing and CI/CD](#testing-and-cicd)
- [Contributing](#contributing)
- [License](#license)

---

## Project Architecture

The application is composed of four main services that work together:

1.  **Streamlit Frontend**: The user-facing web application where users can register, log in, upload potato leaf images for classification, and view their personal prediction history.
2.  **FastAPI Backend**: The core of the application. It handles user authentication, manages user data, processes prediction requests, and interacts with the database. It is the central orchestrator.
3.  **TensorFlow Serving**: A dedicated, high-performance server that hosts the trained CNN model and handles the computational-heavy task of running inference on the uploaded images.
4.  **PostgreSQL Database**: A relational database that stores user credentials and a history of all predictions made by each user.

The typical user flow is as follows:
- A user registers or logs in via the Streamlit interface.
- The FastAPI backend validates the credentials and returns a JWT access token.
- For subsequent requests (like making a prediction), the Streamlit app sends the image and the JWT token to the FastAPI backend.
- The backend validates the token, preprocesses the image, and sends it to the TensorFlow Serving container for a prediction.
- TF Serving returns the prediction result (class and confidence) to the backend.
- The backend saves the prediction result to the PostgreSQL database, linked to the user's ID.
- The result is sent back to the Streamlit app and displayed to the user.

---

## Features

* **Secure User Authentication**: Robust user registration and login system using JWT (JSON Web Tokens) for secure API access. Passwords are securely hashed using `bcrypt`.
* **Accurate Disease Classification**: A Convolutional Neural Network (CNN) distinguishes between "Early Blight," "Late Blight," and "Healthy" potato leaves with high accuracy.
* **Personalized Prediction History**: Authenticated users can view a chronologically sorted history of their past predictions.
* **High-Performance RESTful API**: A fully-featured API built with FastAPI, providing endpoints for user management, authentication, and predictions. Includes automatic interactive documentation (via `/docs`).
* **Interactive Web Interface**: A clean and user-friendly frontend built with Streamlit, enabling easy image uploads and clear presentation of results.
* **Scalable Model Serving**: The TensorFlow model is served using TensorFlow Serving, ensuring low latency and high throughput for predictions.
* **Continuous Integration (CI)**: A GitHub Actions workflow automatically runs tests against the API on every push to the `master` branch, ensuring code quality and stability.
* **Containerized Deployment**: The entire application stack (FastAPI, PostgreSQL) is defined for containerized environments, ensuring consistency and ease of deployment.

---

## Technology Stack

| Category          | Technology                                                                                                  |
| ----------------- | ----------------------------------------------------------------------------------------------------------- |
| **Frontend** | [Streamlit](https://streamlit.io/)                                                                          |
| **Backend** | [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/)                                 |
| **Machine Learning**| [TensorFlow](https://www.tensorflow.org/), [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving), [Pillow](https://pillow.readthedocs.io/en/stable/), [NumPy](https://numpy.org/) |
| **Database** | [PostgreSQL](https://www.postgresql.org/), [SQLAlchemy](https://www.sqlalchemy.org/) (ORM)                  |
| **Authentication**| [python-jose](https://github.com/mpdavis/python-jose) (for JWT), [passlib[bcrypt]](https://passlib.readthedocs.io/en/stable/) (for Hashing) |
| **Data Validation** | [Pydantic](https://docs.pydantic.dev/latest/)                                                               |
| **Testing** | [Pytest](https://docs.pytest.org/), [pytest-mock](https://pytest-mock.readthedocs.io/)                       |
| **Deployment & CI/CD** | [Docker](https://www.docker.com/), [Docker Compose](https://docs.docker.com/compose/), [GitHub Actions](https://github.com/features/actions) |

---

## Project Structure

The repository is organized as follows to maintain a clean separation of concerns:

```
Potato_Disease_Classification/
├── .env                                 # Local environment variables (add to .gitignore)
├── .github/
│   └── workflows/
│       └── ci.yml                       # GitHub Actions CI/CD workflow
├── docker-compose.yml                   # Docker Compose for orchestrating services
├── README.md                            # Project overview and setup instructions
├── Notebook_1.ipynb                     # Jupyter Notebook for model training experiments
├── PlantVillage/                        # Dataset for training
│   ├── Potato___Early_blight/
│   ├── Potato___Late_blight/
│   └── Potato___healthy/
├── Saved_Models/                        # Exported TensorFlow models
│   └── 1/
│       ├── keras_metadata.pb
│       ├── saved_model.pb
│       └── variables/
├── api/                                 # FastAPI backend
│   ├── __pycache__/                     # Python bytecode cache (auto-generated)
│   ├── config.py                        # Pydantic settings and environment config
│   ├── database.py                      # Database connection logic
│   ├── Dockerfile                       # Dockerfile for FastAPI app
│   ├── main_OldVersion.py               # Old app version (can delete or archive)
│   ├── main_Routers.py                  # Main FastAPI app with router inclusion
│   ├── models.py                        # SQLAlchemy ORM models
│   ├── oauth2.py                        # JWT creation/verification logic
│   ├── queries.txt                      # Optional: saved SQL queries or drafts
│   ├── requirements.txt                 # Backend dependencies
│   ├── schemas.py                       # Pydantic request/response schemas
│   ├── tempCodeRunnerFile.py           # Temporary file (safe to delete)
│   ├── utils.py                         # Utility functions (e.g., hashing, token gen)
│   └── routers/                         # API endpoint definitions
│       ├── auth.py                      # Auth endpoints (login, register)
│       ├── prediction.py                # Prediction endpoint (ML model inference)
│       └── user.py                      # User management endpoints
├── streamlit_app/                       # Streamlit frontend app
│   ├── app_AfterAuthorization.py       # Authenticated UI for predictions
│   ├── assets/
│   │   └── farmland.jpg                 # Static image used in frontend
│   └── requirements.txt                 # Streamlit-specific dependencies
├── tests/                               # Pytest-based backend test suite
│   ├── conftest.py                      # Fixtures and setup logic
│   ├── test_predictions.py              # Tests for prediction API
│   ├── test_users.py                    # Tests for user endpoints
│   └── test_utils.py                    # Tests for utility functions

```
---

## Setup and Installation

Follow these steps to get the project running on your local machine.

### **Prerequisites**

* [Docker](https://www.docker.com/products/docker-desktop) and [Docker Compose](https://docs.docker.com/compose/install/) installed and running.
* [Git](https://git-scm.com/) for cloning the repository.
* Python 3.9+ for running the Streamlit app locally.

### **Installation Steps**

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/potato-disease-classification.git](https://github.com/your-username/potato-disease-classification.git)
    cd potato-disease-classification
    ```

2.  **Create the environment file:**
    Create a file named `.env` in the project root directory. Copy the contents of the example below and replace the placeholder values with your own settings.

    **.env.example:**
    ```env
    # Database Configuration for FastAPI and Tests
    DATABASE_HOSTNAME=postgres-db # Use the service name if using Docker Compose
    DATABASE_PORT=5432
    DATABASE_PASSWORD=your_strong_password
    DATABASE_NAME=potato_disease_db
    DATABASE_USERNAME=postgres

    # TensorFlow Serving URL
    # Use 'host.docker.internal' if TF Serving runs on host and API in a container
    # Use the service name (e.g., 'tf-serving') if using Docker Compose
    TF_SERVING_URL=http://tf-serving:8501/v1/models/potato_disease_classifier:predict

    # JWT Authentication Configuration
    SECRET_KEY=generate_a_strong_random_32_byte_hex_string
    ALGORITHM=HS256
    ACCESS_TOKEN_EXPIRE_MINUTES=30
    ```
    **Note**: You can generate a `SECRET_KEY` with `openssl rand -hex 32`.

---

## Running the Application

1. **Launch the Backend Stack:**
    In your terminal, from the project root (`Potato_Disease_Classification`), run the following command:
    ```bash
    docker-compose up --build
    ```
    This single command will build the API image and start all three backend containers (API, Database, Model Server). The API will be available at `http://localhost:8000`.

2.  **Run the Streamlit Frontend:**
    Open a **new terminal window**. Navigate to the project directory and run:
    ```bash
    # (Optional) Create and activate a virtual environment
    # python -m venv venv
    # source venv/bin/activate

    # Install dependencies
    pip install streamlit requests Pillow

    # Run the app on port 8502 to avoid conflict with TF Serving
    streamlit run Potato_Disease_Classification/streamlit_app/app_AfterAuthorization.py --server.port 8502
    ```
    You can now access the web application at **`http://localhost:8502`**.

---

## API Endpoints
The API provides the following endpoints. Protected endpoints require a Bearer token in the Authorization header.

**Authentication**
* POST /login: Authenticates a user and returns a JWT access token.

    * Request Body (form-data):

    ```JSON
    {
        "username": "user@example.com",
        "password": "password123"
    }
    ```

    * Response (200 OK):

    ```JSON
    {
        "access_token": "your.jwt.token",
        "token_type": "bearer"
    }
    ```

**Users** 
* POST /users/: Creates a new user.

    * Request Body:

    ```JSON
    {
        "email": "newuser@example.com",
        "password": "a_strong_password"
    }
    ```

    * Response (201 Created):
    
    ```JSON
    {
        "id": 1,
        "email": "newuser@example.com",
        "created_at": "2025-08-07T12:00:00.000Z"
    }

* GET /users/{id}: Retrieves details for a specific user.

    * Response (200 OK): (Same as user creation response)

***Predictions***
* POST /predictions/ (Protected): Submits an image for classification and saves the result.

    * Request Body (multipart/form-data): An image file (file).
    * Response (200 OK):

    ```JSON
    {
        "id": 1,
        "timestamp": "2025-08-07T12:05:00.000Z",
        "user_id": 1,
        "filename": "leaf.jpg",
        "predicted_class": "Late Blight",
        "confidence": 0.985,
        "owner": {
            "id": 1,
            "email": "user@example.com",
            "created_at": "2025-08-07T12:00:00.000Z"
        }
    }
    
* GET /predictions/ (Protected): Retrieves the prediction history for the authenticated user.

    * Response (200 OK): A list of prediction result objects.

## Machine Learning Model
The classification model is a Convolutional Neural Network (CNN) built and trained using TensorFlow. The entire process is documented in Notebook_1.py.

* Dataset: The model was trained on the PlantVillage dataset, which contains images of healthy and diseased potato leaves across 3 classes.

* Data Preprocessing:

    * Images are resized to a uniform 256x256 pixels.

    * Pixel values are rescaled from [0, 255] to [0, 1] for model stability.

* Data Augmentation: To improve generalization and prevent overfitting, random augmentations like horizontal/vertical flips and rotations are applied to the training data in real-time.

* Model Architecture: The CNN consists of multiple convolutional and max-pooling layers to extract features, followed by dense layers for classification.

    * 6 Conv2D layers with relu activation.

    * 6 MaxPooling2D layers for down-sampling.

    * A Flatten layer to transition to the classifier head.

    * A Dense layer with 64 units (relu activation).

    * The final Dense output layer with 3 units and softmax activation for multi-class probability distribution.

* Training & Performance:

    * The model was trained for 40 epochs using the Adam optimizer and SparseCategoricalCrossentropy loss function.

    * The final model achieved an accuracy of approximately 85.5% on the hold-out test set.

---

## Testing and CI/CD
The project emphasizes code quality through a comprehensive testing suite and a full CI/CD pipeline.

***Testing***

* Framework: Pytest is used for writing and running tests.

* Test Database: Tests run against a separate, isolated test database (..._test) to avoid interfering with development data. The database is reset for each test function.

* Fixtures (conftest.py):

    * session: Provides a clean database session for each test.
    * client: Creates a TestClient for the FastAPI app.
    * test_user: Creates and returns a new user in the test database.
    * authorized_client: Provides a TestClient with pre-set authorization headers for testing protected endpoints.

* Coverage: Tests cover user creation, user login (valid and invalid), password hashing, and prediction endpoints (unauthorized, successful prediction, and history retrieval).

**Continuous Integration & Delivery (CI/CD)**
* **Platform**: GitHub Actions.
* **Workflow (`ci.yml`)**:
    1.  **Trigger**: The workflow runs automatically on every `push` or `pull_request` to the `master` branch.
    2.  **Test Job**: It spins up a temporary PostgreSQL container, installs all dependencies, and runs the entire `pytest` suite to validate the code.
    3.  **Build & Push Job**: If the tests pass, the workflow logs into Docker Hub, builds a new Docker image for the API, and pushes it with the `:latest` tag.
    4.  **Deploy Job**: A placeholder `deploy` job demonstrates where deployment scripts to a production server would be placed. It runs only after a successful build on a push to `master`.

---

## Contributing
Contributions are welcome! If you have any ideas, suggestions, or bug reports, please feel free to open an issue or submit a pull request.

1. Fork the Project.

2. Create your Feature Branch (git checkout -b feature/AmazingFeature).

3. Commit your Changes (git commit -m 'Add some AmazingFeature').

4. Push to the Branch (git push origin feature/AmazingFeature).

5. Open a Pull Request.

---

## License
This project is licensed under the MIT License - see the LICENSE file for details.


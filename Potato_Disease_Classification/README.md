# Potato Disease Classification Full-Stack Project

This project is a full-stack application designed to classify potato diseases from leaf images. It utilizes a deep learning model trained with TensorFlow, served via TensorFlow Serving, and exposed through a FastAPI backend. The user interface is a web application built with Streamlit, allowing users to upload potato leaf images and receive a diagnosis. All components of the application are containerized using Docker for easy deployment and scalability.

---

## Features

* **Accurate Disease Classification**: Distinguishes between "Early Blight," "Late Blight," and "Healthy" potato leaves.
* **Interactive User Interface**: A user-friendly web interface built with Streamlit for easy image uploads and viewing results.
* **High-Performance API**: A robust FastAPI backend serves predictions, with the heavy lifting of inference handled by TensorFlow Serving.
* **Prediction History**: All prediction results are stored in a PostgreSQL database for historical tracking.
* **Dockerized Deployment**: The entire application stack (FastAPI, TensorFlow Serving, and database) is containerized for seamless setup and deployment.

---

## Technologies Used

### **Frontend**

* [Streamlit](https://streamlit.io/): For creating the interactive web application.

### **Backend**

* [FastAPI](https://fastapi.tiangolo.com/): As the web framework for the prediction API.
* [PostgreSQL](https://www.postgresql.org/): For storing prediction history.
* [SQLAlchemy](https://www.sqlalchemy.org/): As the Object-Relational Mapper (ORM) for database interaction.
* [Pydantic](https://docs.pydantic.dev/): For data validation and settings management.

### **Machine Learning**

* [TensorFlow](https://www.tensorflow.org/): For building and training the deep learning model.
* [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving): For serving the trained model in a production environment.
* [Pillow](https://pillow.readthedocs.io/en/stable/): For image manipulation.
* [NumPy](https://numpy.org/): For numerical operations.
* [Matplotlib](https://matplotlib.org/): For data visualization during model development.

### **Deployment**

* [Docker](https://www.docker.com/): For containerizing the application components.
* [Uvicorn](https://www.uvicorn.org/): As the ASGI server for FastAPI.

---

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### **Prerequisites**

* [Docker Desktop](https://www.docker.com/products/docker-desktop) installed and running on your system.
* A Git client to clone the repository.

### **Setup and Installation**

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/potato-disease-classification.git](https://github.com/your-username/potato-disease-classification.git)
    cd potato-disease-classification
    ```
2.  **Project Structure:** The project is organized into the following main directories:
    * `api/`: Contains the FastAPI application, database models, and Dockerfile for the backend.
    * `streamlit_app/`: Contains the Streamlit application for the user interface.
    * `Saved_Models/`: Stores the trained TensorFlow model.
    * `PlantVillage/`: Contains the dataset used for training the model.

---

## Running the Application

The application is designed to be run using Docker.

### **1. Start TensorFlow Serving**

Open a terminal and run the following command to start the TensorFlow Serving container. This command mounts the saved model from your local machine into the container.

```bash
docker run -p 8501:8501 --mount type=bind,source="C:\Users\ashut\ML_Projects\Potato_Disease_Classification\Saved_Models",target=/models/potato_disease_classifier -e MODEL_NAME=potato_disease_classifier -t tensorflow/serving
```

### **2. Build and Run the FastAPI Application**
In a new terminal, navigate to the api directory and run the following commands to build and start the FastAPI container.

```bash
cd api
docker build -t fastapi-potato-classifier .
docker run -d --name fastapi_potato_app -p 8000:8000 fastapi-potato-classifier
```

### **3. Run the Streamlit Application**
Finally, in another terminal, navigate to the streamlit_app directory and run the Streamlit application.

```bash
cd ../streamlit_app
streamlit run app.py --server.port 8502
```

You can now access the Streamlit web application by navigating to http://localhost:8502 in your web browser.

## API Endpoints
The FastAPI application provides the following endpoints:

* GET /ping: A simple health check endpoint.

  * Response: "Hello, I am alive"

* POST /predict: Accepts an image file and returns the predicted class and confidence.

  * Request Body: UploadFile (image file)

  * Response:

JSON
```bash
{
  "class": "Predicted_Class",
  "confidence": 99.99
}
```

* GET /history: Retrieves a list of all past predictions from the database.

  * Response: A JSON array of prediction results.

JSON
```bash
[
    {
        "id": 1,
        "filename": "image.jpg",
        "predicted_class": "Early_Blight",
        "confidence": 0.9876,
        "timestamp": "2025-07-11T10:00:00Z"
    }
]
```
---

## Database
The application uses a PostgreSQL database to store the history of all predictions. The PredictionResult model in models.py defines the schema for the prediction_results table, which includes the following fields:

* id: Primary key.

* filename: The name of the uploaded image file.

* predicted_class: The predicted class of the potato disease.

* confidence: The confidence score of the prediction.

* timestamp: The timestamp of when the prediction was made.

---

## Contributing
Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

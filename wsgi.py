from applr import app
from applr import app_lr

if __name__ == "__main__":
    app.run(port=5000)  # Run the first Flask app
    app_lr.run(port=5001)  # Run the second Flask app on a different port

# Import the function that creates the Flask app instance
from app import create_app
import os # Import os for environment variables if needed

# Optional: Load environment variables or configurations if needed
# For example, using python-dotenv:
# from dotenv import load_dotenv
# load_dotenv()

# Call the factory function to create the app instance
app = create_app()

# The rest of your main.py logic might go here,
# although often this file is just used for deployment runners like Gunicorn.
# If you were using Gunicorn, it would typically point directly to 'app:app'
# where the second 'app' refers to the variable created by calling create_app().

# If you intend to run the app directly using `python main.py` (for development),
# you might add this block, similar to app.py:
if __name__ == '__main__':
    # You might want to configure host and port via environment variables
    port = int(os.environ.get('PORT', 5000))
    # Use debug=False for production/deployment
    # Use debug=True only for development
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() in ['true', '1', 't']
    app.run(debug=debug_mode, host='0.0.0.0', port=port)



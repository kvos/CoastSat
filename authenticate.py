import ee
#script to run to test ee connexion before running example scripts
def authenticate_and_initialize():
    try:
        # Authenticate the Earth Engine session.
        ee.Authenticate()
        # Initialize the Earth Engine module.
        ee.Initialize()
        print("Authentication successful!")
    except Exception as e:
        print(f"Authentication failed: {e}")

if __name__ == "__main__":
    authenticate_and_initialize()

from app_enhanced import app

# Export app for gunicorn
# gunicorn will use: gunicorn main:app
if __name__ == "__main__":
    import sys
    port = 5001  # Default port
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port number: {sys.argv[1]}. Using default port 5001.")
    
    app.run(host='0.0.0.0', port=port, debug=False)
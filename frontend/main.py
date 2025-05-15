import http.server
import socketserver
import os

PORT = 8000
# Serve files from the directory where the script is run.
# This assumes index.html and app.js are in the same directory as static_server.py
DIRECTORY_TO_SERVE = "./frontend"  # Change this to the directory you want to serve

class SimpleHTTPRequestHandlerWithCORS(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Set the directory to serve before calling the superclass constructor
        super().__init__(*args, directory=DIRECTORY_TO_SERVE, **kwargs)

    def end_headers(self):
        # Add CORS headers to allow requests from any origin,
        # which might be useful if index.html is opened from a different port
        # or if you decide to serve index.html from a different origin later.
        # For a production environment, you'd want to restrict this.
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        # Respond to preflight CORS requests
        self.send_response(200)
        self.end_headers()

if __name__ == "__main__":
    # Ensure the server runs from the script's directory if needed,
    # but by default, DIRECTORY_TO_SERVE = "." serves from the CWD.
    # If you always want to serve from the script's location:
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # os.chdir(script_dir)
    # print(f"Serving files from directory: {os.path.abspath(script_dir)}")
    
    # Get the absolute path of the directory to be served
    abs_serve_directory = os.path.abspath(DIRECTORY_TO_SERVE)

    with socketserver.TCPServer(("", PORT), SimpleHTTPRequestHandlerWithCORS) as httpd:
        print(f"Serving files from directory: {abs_serve_directory}")
        print(f"Server started at http://localhost:{PORT}/ or http://0.0.0.0:{PORT}/")
        print("Press Ctrl+C to stop the server.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer is shutting down.")
            httpd.shutdown()
            print("Server stopped.")

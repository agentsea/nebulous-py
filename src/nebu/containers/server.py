import base64
import json
import pickle
from http.server import BaseHTTPRequestHandler, HTTPServer


class FunctionExecutionHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        payload = json.loads(post_data.decode("utf-8"))

        if self.path == "/execute":
            try:
                # Extract function code, args and kwargs
                func_code = payload["function_code"]
                serialized_args = payload["args"]
                serialized_kwargs = payload["kwargs"]

                # Deserialize arguments
                args = pickle.loads(base64.b64decode(serialized_args))
                kwargs = pickle.loads(base64.b64decode(serialized_kwargs))

                # Create a local namespace and execute the function
                local_namespace = {}
                exec(func_code, globals(), local_namespace)

                # Find the function object in the local namespace
                func_name = None
                for name, obj in local_namespace.items():
                    if callable(obj) and not name.startswith("__"):
                        func_name = name
                        break

                if not func_name:
                    raise ValueError("No function found in the provided code")

                # Execute the function
                result = local_namespace[func_name](*args, **kwargs)

                # Serialize the result
                serialized_result = base64.b64encode(pickle.dumps(result)).decode(
                    "utf-8"
                )

                # Send response
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"result": serialized_result}).encode())

            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()


def run_server(port: int = 8080) -> None:
    server_address = ("", port)
    httpd = HTTPServer(server_address, FunctionExecutionHandler)
    print(f"Starting server on port {port}")
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()

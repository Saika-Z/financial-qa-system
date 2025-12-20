
import subprocess
import os
import platform
import sys
import signal


def start_server():
    #1. find path
    root_dir = os.path.abspath(os.path.dirname(__file__))

    # 2.
    backend_module = "backend.app.main"
    frontend_path = os.path.join(root_dir, 'frontend')


    # 3. 
    python_executable = sys.executable


    print(f"--- 🚀 starting financial-qa-system server on {platform.system()}---")

    process = []

    try:
        print(f"--- 📦 starting backend server on {backend_module}---")
        backend_proc = subprocess.Popen(
            [python_executable, "-m" , backend_module],
            cwd=root_dir
        )
        process.append(backend_proc)

        print(f"--- 🎨 starting frontend server on {frontend_path}---")
        is_windows = platform.system() == 'Windows'
        fortend_proc = subprocess.Popen(
            ["npm", "run", "serve"],
            cwd=frontend_path,
            shell=is_windows
        )
        process.append(fortend_proc)

        print("\n server started successfully.")
        print(" Press Ctrl+C to stop the server. ")

        for p in process:
            p.wait()

    except KeyboardInterrupt:
        print("\n quit signal received, shutting down...")
        for p in process:
            p.terminate()
        print("👋 all processes terminated.")


if __name__ == '__main__':
    start_server()
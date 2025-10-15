import subprocess
import time
import threading
import requests
import sys

def check_server_ready():
    """Check if Flask server is responding"""
    for i in range(30):  # Try for 30 seconds
        try:
            response = requests.get("http://localhost:5000", timeout=5)
            if response.status_code == 200:
                print("✓ Flask server is ready!")
                return True
        except:
            print(f"Waiting for server... ({i+1}/30)")
            time.sleep(1)
    return False

def start_flask():
    """Start Flask app"""
    subprocess.run([sys.executable, "app.py"])

print("=" * 50)
print("    FRAUD DETECTION SYSTEM - PUBLIC")
print("=" * 50)
print()

# Start Flask
print("Starting Flask server...")
flask_thread = threading.Thread(target=start_flask)
flask_thread.daemon = True
flask_thread.start()

# Wait for server to be ready
if check_server_ready():
    print("\n" + "=" * 50)
    print("SERVER IS READY!")
    print("Now run this command in a NEW command window:")
    print("ssh -R 80:localhost:5000 nokey@localhost.run")
    print("=" * 50)
    print("\nKeep this window open and follow the steps above.")
    input("Press Enter when you've started the tunnel...")
else:
    print("❌ Server failed to start in time")
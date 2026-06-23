 import subprocess
 import pytest

 def test_ollama_running():
     try:
         # Run the ollama command and capture output and return code
         result = subprocess.run(['ollama', '--version'], check=True, text=True, capture_output=True)
         
         # Check if the version string is in the stdout (this assumes that --version outputs a valid version string)
         assert "ollama" in result.stdout, "The ollama command did not output the expected version string."
     except subprocess.CalledProcessError as e:
         pytest.fail(f"The ollama command failed with exit code {e.returncode}. Output: {e.output}")

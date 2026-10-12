// Simple test to check if backend is accessible
fetch('http://localhost:8000/health')
  .then(response => response.json())
  .then(data => console.log('Backend response:', data))
  .catch(error => console.error('Backend error:', error));

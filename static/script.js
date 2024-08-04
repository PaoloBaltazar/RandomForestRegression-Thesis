document.getElementById('prediction-form').addEventListener('submit', function(event) {
  event.preventDefault();

  const formData = new FormData(event.target);
  const data = {
      features: [
          parseFloat(formData.get('attendance')),
          parseFloat(formData.get('financial_situation')),
          parseFloat(formData.get('learning_environment')),
          parseFloat(formData.get('previous_grades'))
      ]
  };

  fetch('/predict', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
  })
  .then(response => response.json())
  .then(data => {
      const resultDiv = document.getElementById('prediction-result');
      resultDiv.innerHTML = `Predicted Grade: ${data.prediction}`;
  })
  .catch(error => console.error('Error:', error));
  
});
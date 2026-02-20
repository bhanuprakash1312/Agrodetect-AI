const API_BASE_URL = 'https://agriscan-3b2j.onrender.com'; // Update with your actual API base URL

export interface PredictionResponse {
  prediction: string;
  treatment: string;
}

export const verifyLeaf = async (imageFile: File): Promise<boolean> => {
  const formData = new FormData();
  formData.append('file', imageFile);

  const response = await fetch(`${API_BASE_URL}/verify-leaf`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    console.warn('Leaf verification failed:', error.detail);
    return false;
  }

  return true;
};


export const predictDisease = async (imageFile: File): Promise<PredictionResponse | { detail: string }> => {
  const formData = new FormData();
  formData.append('file', imageFile);


  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    body: formData,
  });


  // Always parse response as JSON (even for errors)
  const result = await response.json();

  // Handle non-leaf rejection from backend
  if (!response.ok) {
    return result; // May contain: { detail: "Uploaded image is not a leaf." }
  }

  // Return disease prediction
  const mappedResult: PredictionResponse = {
    prediction: result.disease || result.prediction || 'Unknown disease',
    treatment: result.treatment || 'No treatment available',
  };

  return mappedResult;
};

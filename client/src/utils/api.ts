// const API_BASE_URL = 'https://agriscan-3b2j.onrender.com'; 
const API_BASE_URL = 'https://agrodetect-ai-production.up.railway.app';

export interface PredictionResponse {
  prediction: string;
  treatment: string;
}

export interface TranslationResponse {
  prediction: string;
  treatment: string;
}

export interface ActionsResponse {
  actions: string[];
  prevention: string[];
}


// Existing leaf verification (UNCHANGED)
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


// Existing prediction (UNCHANGED)
export const predictDisease = async (
  imageFile: File
): Promise<PredictionResponse | { detail: string }> => {

  const formData = new FormData();
  formData.append('file', imageFile);

  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    body: formData,
  });

  const result = await response.json();

  if (!response.ok) {
    return result;
  }

  const mappedResult: PredictionResponse = {
    prediction: result.disease || result.prediction || 'Unknown disease',
    treatment: result.treatment || 'No treatment available',
  };

  return mappedResult;
};


//////////////////////////////////////////////////////
// NEW: Translation function
//////////////////////////////////////////////////////

export const translateDisease = async (
  prediction: string,
  treatment: string,
  language: string
): Promise<TranslationResponse> => {

  const formData = new FormData();

  formData.append("prediction", prediction);
  formData.append("treatment", treatment);
  formData.append("language", language);

  const response = await fetch(`${API_BASE_URL}/translate`, {
    method: "POST",
    body: formData,
  });

  const result = await response.json();

  return {
    prediction: result.prediction,
    treatment: result.treatment,
  };
};


//////////////////////////////////////////////////////
// NEW: Get Gemini actions and prevention
//////////////////////////////////////////////////////

export const getActionsAndPrevention = async (
  disease: string,
  language: string
): Promise<ActionsResponse> => {

  const formData = new FormData();

  formData.append("disease", disease);
  formData.append("language", language);

  const response = await fetch(`${API_BASE_URL}/generate-actions`, {
    method: "POST",
    body: formData,
  });

  const result = await response.json();

  return {
    actions: result.actions || [],
    prevention: result.prevention || [],
  };
};
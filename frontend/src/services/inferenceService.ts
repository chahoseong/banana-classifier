export interface PredictionResult {
  is_banana: boolean;
  status: 'unripe' | 'ripe' | 'overripe' | 'dispose' | 'none';
  confidence: number;
  message: string;
}

const API_BASE_URL = 'http://127.0.0.1:8000'; // Use IP instead of localhost for Windows reliability

export async function predictBanana(imageBlob: Blob, modelId: string = 'baseline'): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append('file', imageBlob, 'capture.jpg');
  formData.append('model_id', modelId);

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Server responded with ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Prediction API Error:', error);
    throw error;
  }
}
export interface ModelInfo {
  id: string;
  name: string;
  description: string;
  type: string;
}

export async function getAvailableModels(): Promise<ModelInfo[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/models`);
    if (!response.ok) {
      throw new Error(`Failed to fetch models: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Fetch Models Error:', error);
    throw error;
  }
}

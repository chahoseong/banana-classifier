export interface PredictionResult {
  is_banana: boolean;
  status: 'unripe' | 'ripe' | 'overripe' | 'dispose' | 'none';
  confidence: number;
  message: string;
}

const API_BASE_URL = 'http://127.0.0.1:8000'; // Use IP instead of localhost for Windows reliability

export async function predictBanana(imageBlob: Blob): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append('file', imageBlob, 'capture.jpg');

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

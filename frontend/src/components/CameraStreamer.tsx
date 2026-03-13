import React, { useRef, useEffect, useState, useCallback } from 'react';
import { predictBanana, PredictionResult } from '../services/inferenceService';

interface CameraStreamerProps {
  onResult: (result: PredictionResult) => void;
  onError: (error: string) => void;
  modelId: string;
}

const CameraStreamer: React.FC<CameraStreamerProps> = ({ onResult, onError, modelId }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);

  // 1. 카메라 시작
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: 640, height: 480 },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsStreaming(true);
      }
    } catch (err) {
      console.error('Camera Access Error:', err);
      onError('카메라에 접근할 수 없습니다. 권한을 확인해 주세요.');
    }
  };

  // 2. 캡처 및 전송 로직 (Throttle & Recursive)
  const processFrame = useCallback(async () => {
    if (!isStreaming || isProcessing || !videoRef.current || !canvasRef.current) return;

    setIsProcessing(true);
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    const video = videoRef.current;

    if (context) {
      // 300x300 해상도로 캡처 (중앙 크롭)
      const size = Math.min(video.videoWidth, video.videoHeight);
      const startX = (video.videoWidth - size) / 2;
      const startY = (video.videoHeight - size) / 2;

      context.drawImage(video, startX, startY, size, size, 0, 0, 300, 300);

      canvas.toBlob(async (blob) => {
        if (blob) {
          try {
            const result = await predictBanana(blob, modelId);
            onResult(result);
            
            // 협업 가이드 가이드: 성공 후 500ms 대기 후 다음 자동 호출
            setTimeout(() => {
              setIsProcessing(false);
            }, 500);
          } catch (err) {
            onError('분석 서버 통신 중 오류가 발생했습니다. 재시도합니다.');
            // 에러 시 2초 후 재시도
            setTimeout(() => {
              setIsProcessing(false);
            }, 2000);
          }
        }
      }, 'image/jpeg', 0.8);
    }
  }, [isStreaming, isProcessing, onResult, onError, modelId]);

  useEffect(() => {
    startCamera();
    return () => {
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    if (isStreaming && !isProcessing) {
      processFrame();
    }
  }, [isStreaming, isProcessing, processFrame]);

  return (
    <div className="relative w-full max-w-[500px] aspect-square bg-black rounded-3xl overflow-hidden shadow-2xl border-4 border-slate-700">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="absolute inset-0 w-full h-full object-cover"
      />
      <canvas ref={canvasRef} width={300} height={300} className="hidden" />
      
      {/* UI Guide: ROI Square */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <div className="w-3/4 h-3/4 border-2 border-white/50 rounded-2xl border-dashed animate-pulse">
        </div>
      </div>
      
      {isProcessing && (
        <div className="absolute top-4 right-4 flex items-center gap-2 bg-black/60 px-3 py-1 rounded-full text-xs text-white backdrop-blur-sm">
          <div className="w-2 h-2 bg-yellow-400 rounded-full animate-ping" />
          분석 중...
        </div>
      )}
    </div>
  );
};

export default CameraStreamer;

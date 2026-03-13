import React from 'react';
import { PredictionResult } from '../services/inferenceService';

interface ResultDisplayProps {
  result: PredictionResult | null;
  error: string | null;
}

const statusConfig = {
  unripe: { color: 'border-banana-unripe', text: 'border-banana-unripe', bg: 'bg-banana-unripe/10', label: '덜 익음 (Unripe)' },
  ripe: { color: 'border-banana-ripe', text: 'text-banana-ripe', bg: 'bg-banana-ripe/10', label: '잘 익음 (Ripe)' },
  overripe: { color: 'border-banana-overripe', text: 'text-banana-overripe', bg: 'bg-banana-overripe/10', label: '너무 익음 (Overripe)' },
  dispose: { color: 'border-banana-dispose', text: 'text-banana-dispose', bg: 'bg-banana-dispose/10', label: '폐기 권장 (Dispose)' },
  none: { color: 'border-slate-500', text: 'text-slate-500', bg: 'bg-slate-500/10', label: '인식 중...' },
  checking: { color: 'border-blue-500', text: 'text-blue-500', bg: 'bg-blue-500/10', label: '상태 분석 중...' },
};

const ResultDisplay: React.FC<ResultDisplayProps> = ({ result, error }) => {
  if (error) {
    return (
      <div className="w-full mt-6 bg-red-900/20 border border-red-500/50 p-4 rounded-2xl text-red-400 text-sm animate-bounce">
        ⚠️ {error}
      </div>
    );
  }

  if (!result) {
    return (
      <div className="w-full mt-6 bg-slate-800/50 border border-slate-700 p-8 rounded-2xl text-center text-slate-400 italic">
        바나나를 카메라 중앙에 위치시켜 주세요
      </div>
    );
  }

  const currentStatus = (result.is_banana ? result.status : 'none') || 'none';
  const config = statusConfig[currentStatus as keyof typeof statusConfig] || statusConfig.none;

  return (
    <div className={`w-full mt-6 border-2 ${config.color} ${config.bg} p-6 rounded-2xl transition-all duration-500 ease-in-out shadow-lg`}>
      <div className="flex justify-between items-center mb-4">
        <span className={`text-2xl font-black uppercase tracking-tighter ${config.text}`}>
          {result.is_banana ? config.label : '바나나가 아님'}
        </span>
        <span className="text-xs bg-black/40 px-2 py-1 rounded font-mono text-slate-300">
          CONFIDENCE: {(result.confidence * 100).toFixed(1)}%
        </span>
      </div>
      <p className="text-slate-200 text-lg leading-snug font-medium">
        {result.message}
      </p>
      
      {/* Progress bar visual for confidence */}
      <div className="mt-4 w-full bg-black/20 h-1.5 rounded-full overflow-hidden">
        <div 
          className={`h-full transition-all duration-1000 bg-current ${config.text}`}
          style={{ width: `${result.confidence * 100}%` }}
        />
      </div>
    </div>
  );
};

export default ResultDisplay;

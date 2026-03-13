import { useState } from 'react';
import CameraStreamer from './components/CameraStreamer';
import ResultDisplay from './components/ResultDisplay';
import { PredictionResult } from './services/inferenceService';
import { Banana, Scan, Info } from 'lucide-react';

function App() {
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleResult = (newResult: PredictionResult) => {
    setResult(newResult);
    setError(null);
  };

  const handleError = (errorMessage: string) => {
    setError(errorMessage);
  };

  return (
    <div className="min-h-screen w-full bg-[#0f172a] text-slate-200 font-sans selection:bg-yellow-500/30 flex flex-col items-center">
      {/* Header */}
      <header className="w-full border-b border-white/5 bg-slate-900/50 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-3xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-yellow-500 p-2 rounded-xl shadow-lg shadow-yellow-500/20">
              <Banana className="text-slate-900 w-6 h-6" />
            </div>
            <div>
              <h1 className="text-xl font-black tracking-tight text-white leading-none">BANANA</h1>
              <span className="text-[10px] font-bold text-yellow-500 uppercase tracking-widest px-0.5">Ripe Checker</span>
            </div>
          </div>
          <button className="text-slate-500 hover:text-white transition-colors">
            <Info className="w-5 h-5" />
          </button>
        </div>
      </header>

      <main className="w-full max-w-3xl mx-auto px-6 py-10 flex-grow">
        {/* Intro Section */}
        <section className="mb-10 text-center">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-yellow-500/10 border border-yellow-500/20 text-yellow-500 text-xs font-bold mb-4">
            <Scan className="w-3 h-3" />
            REAL-TIME AI ANALYSIS
          </div>
          <h2 className="text-3xl font-bold text-white mb-2">바나나 숙성도 판별</h2>
          <p className="text-slate-400 text-sm">
            카메라에 바나나를 비추면 AI가 실시간으로 신선도를 분석합니다.
          </p>
        </section>

        {/* Camera Section */}
        <div className="flex flex-col items-center">
          <CameraStreamer onResult={handleResult} onError={handleError} />
          
          {/* Result Display */}
          <ResultDisplay result={result} error={error} />
        </div>

        {/* Support Info */}
        <footer className="mt-20 pt-10 border-t border-white/5 text-center px-4">
          <p className="text-slate-500 text-xs leading-relaxed max-w-sm mx-auto">
            이 모델은 kNN 알고리즘을 사용하여 색상 분포를 분석합니다.<br/>
            조명 환경에 따라 결과가 달라질 수 있으니 밝은 곳에서 사용해 주세요.
          </p>
          <div className="mt-8 flex justify-center gap-6">
            <div className="w-8 h-8 rounded-full bg-banana-unripe/20 border border-banana-unripe/30" title="Unripe" />
            <div className="w-8 h-8 rounded-full bg-banana-ripe/20 border border-banana-ripe/30" title="Ripe" />
            <div className="w-8 h-8 rounded-full bg-banana-overripe/20 border border-banana-overripe/30" title="Overripe" />
            <div className="w-8 h-8 rounded-full bg-banana-dispose/20 border border-banana-dispose/30" title="Dispose" />
          </div>
        </footer>
      </main>
    </div>
  );
}

export default App;

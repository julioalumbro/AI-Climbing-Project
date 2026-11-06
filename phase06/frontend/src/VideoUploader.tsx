import { useState } from "react";

export interface UploadResult {
  upload_id: string;
  filename: string;
  saved_filename: string;
  size_bytes: number;
  status: string;
}


export type HybridMetricEvaluation = {
  metric: string;
  status: string;
  message: string;
};

export type HybridFeedbackItem = {
  movement_type: string;
  start_seconds: number;
  end_seconds: number;
  observation: string;
  gemini_confidence: string;
  evidence_status: string;
  metric_evaluations: HybridMetricEvaluation[];
  feedback: string;
};

export interface AnalysisResult {
  upload_id: string;
  status: string;
  frames_processed: number;
  left_arm_summary: Record<string, number | null>;
  right_arm_summary: Record<string, number | null>;
  metrics_url: string;
  annotated_video_url: string;
  feedback: CoachingFeedback[];
  hybrid_status?: string;
  hybrid_error?: string | null;
  hybrid_feedback?: HybridFeedbackItem[];
}

export interface CoachingFeedback {
  type: string;
  body_part: string;
  start_seconds: number;
  end_seconds: number;
  duration_seconds: number;
  observation: string;
  interpretation: string;
  recommendation: string;
  evidence_status: string;
  limitations: string[];

}

export interface GeminiCoaching {
  overall_assessment: string;
  strengths: string[];
  improvements: string[];
  limitations: string[];
}

interface VideoUploaderProps {
  onVideoSelected: (file: File | null) => void;
  onAnalysisComplete: (analysis: AnalysisResult) => void;
  onReset: () => void;
}

async function responseError(response: Response, fallback: string): Promise<string> {
  try {
    const data: { detail?: string } = await response.json();
    return data.detail || fallback;
  } catch {
    return fallback;
  }
}

export default function VideoUploader({ onVideoSelected, onAnalysisComplete, onReset }: VideoUploaderProps) {
  const [video, setVideo] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<UploadResult | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const busy = uploading || analyzing;

  async function handleUpload() {
    if (!video || busy) return;
    setUploading(true);
    setError(null);
    setResult(null);
    setAnalysis(null);
    onReset();
    try {
      const formData = new FormData();
      formData.append("file", video);
      const response = await fetch("/api/upload", { method: "POST", body: formData });
      if (!response.ok) throw new Error(await responseError(response, "Upload failed"));
      const data: UploadResult = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected upload error");
    } finally {
      setUploading(false);
    }
  }

  async function handleAnalyze() {
    if (!result || busy || analysis) return;
    setAnalyzing(true);
    setError(null);
    try {
      const response = await fetch(`/api/analyze/${result.upload_id}`, { method: "POST" });
      if (!response.ok) throw new Error(await responseError(response, "Analysis failed"));
      const data: AnalysisResult = await response.json();
      if (data.status !== "completed") throw new Error("Analysis did not complete");
      setAnalysis(data);
      onAnalysisComplete(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected analysis error");
    } finally {
      setAnalyzing(false);
    }
  }

  return (
    <section className="rounded-xl border border-slate-700 p-5 space-y-4">
      <h2 className="text-xl font-semibold">Upload Climbing Video</h2>
      <div className="flex items-center gap-3">
        <label
          className={`inline-flex w-fit items-center gap-2 rounded-lg
            bg-lime-400 px-3 py-1.5 text-xs font-semibold
            text-slate-950 transition-colors
            ${busy ? "cursor-not-allowed opacity-50" : "cursor-pointer hover:bg-lime-300"}`}
        >
          Select Video

          <input
            type="file"
            accept=".mp4,.mov,.webm"
            className="sr-only"
            disabled={busy}
            onChange={(event) => {
              const file = event.target.files?.[0] ?? null;
              setVideo(file);
              setResult(null);
              setAnalysis(null);
              setError(null);
              onReset();
              onVideoSelected(file);
            }}
          />
        </label>
        <span className="text-sm text-slate-400">
          {video ? video.name : "No file selected"}
        </span>
      </div>
      <div className="flex flex-wrap gap-3">
        <button type="button" onClick={handleUpload} disabled={!video || busy || !!result}
          className="rounded-lg bg-blue-600 px-4 py-2 text-white disabled:opacity-50">
          {uploading ? "Uploading..." : result ? "Uploaded" : "Upload Video"}
        </button>
        <button type="button" onClick={handleAnalyze} disabled={!result || busy || !!analysis}
          className="rounded-lg bg-lime-400 px-4 py-2 font-semibold text-slate-950 disabled:opacity-50">
          {analyzing ? "Analyzing... please wait" : analysis ? "Analysis complete" : "Analyze Video"}
        </button>
      </div>
      {analyzing && <p className="text-sm text-amber-300" role="status">MediaPipe is processing the video and FFmpeg is converting it. This can take a few minutes. Keep this page open.</p>}
      {error && <p className="text-red-400" role="alert">Error: {error}</p>}
      {result && (
        <div className="rounded-lg bg-slate-800 p-4 space-y-2">
          <p className="text-green-400">Video uploaded successfully!</p>
          <p>Filename: {result.filename}</p>
          <p>Size: {(result.size_bytes / 1024 / 1024).toFixed(2)} MB</p>
          <p className="break-all text-xs text-slate-400">Upload ID: {result.upload_id}</p>
          {analysis && <p className="text-lime-300">Analysis complete: {analysis.frames_processed.toLocaleString()} frames processed.</p>}
        </div>
      )}
    </section>
  );
}

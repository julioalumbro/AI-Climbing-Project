import { useEffect, useRef, useState } from 'react';
import { Activity, ArrowUpRight, BrainCircuit, CheckCircle2, Clapperboard, Gauge, Info, Mountain, ShieldCheck, Video } from 'lucide-react';
import VideoUploader, { type AnalysisResult } from './VideoUploader';
import type { GeminiCoaching} from "./VideoUploader";

type Health = { status: string; service: string; phase: string };
type VideoMode = 'original' | 'skeleton';

function seconds(value: number | null | undefined) {
  return typeof value === 'number' && Number.isFinite(value) ? `${value.toFixed(2)}s` : 'N/A';
}

export default function App() {
  const [health, setHealth] = useState<'checking' | 'connected' | 'offline'>('checking');
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState('');
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const playbackRef = useRef({time: 0, wasPlaying: true,});
  const [videoMode, setVideoMode] = useState<VideoMode>('original');

  useEffect(() => {
    const controller = new AbortController();
    fetch('/api/health', { signal: controller.signal })
      .then(async (r) => { if (!r.ok) throw new Error('API error'); return r.json() as Promise<Health>; })
      .then((data) => setHealth(data.status === 'ok' ? 'connected' : 'offline'))
      .catch((e: unknown) => { if (!(e instanceof DOMException && e.name === 'AbortError')) setHealth('offline'); });
    return () => controller.abort();
  }, []);

  useEffect(() => () => { if (videoUrl) URL.revokeObjectURL(videoUrl); }, [videoUrl]);


function selectVideo(file: File | null) {
  setAnalysis(null);
  setVideoMode('original');
  playbackRef.current = { time: 0, wasPlaying: false };

  // Clear previous Gemini coaching
  setGeminiCoaching(null);
  setCoachingError(null);

  if (!file) {
    setVideoUrl(null);
    setFileName('');
    return;
  }

  setVideoUrl(URL.createObjectURL(file));
  setFileName(file.name);
}


  const annotatedUrl = analysis ? analysis.annotated_video_url : null;
  const activeVideoUrl = videoMode === 'skeleton' && annotatedUrl ? annotatedUrl : videoUrl;
  const metrics = analysis ? [
    { label: 'Frames processed', value: analysis.frames_processed.toLocaleString(), note: 'Frames analyzed by MediaPipe' },
    { label: 'Left arm · longest deep bend', value: seconds(analysis.left_arm_summary.longest_duration), note: 'Longest detected deep-bend interval' },
    { label: 'Left arm · total deep bend', value: seconds(analysis.left_arm_summary.total_duration), note: 'Total detected deep-bend time' },
    { label: 'Right arm · longest deep bend', value: seconds(analysis.right_arm_summary.longest_duration), note: 'Longest detected deep-bend interval' },
    { label: 'Right arm · total deep bend', value: seconds(analysis.right_arm_summary.total_duration), note: 'Total detected deep-bend time' },
  ] : [];

  const jumpToTimestamp = (seconds: number) => {
  const video = videoRef.current;

    if (!video) {
      return;
    }

    video.currentTime = seconds;
  };

  const switchVideoMode = (mode: 'original' | 'skeleton') => {
  const video = videoRef.current;

  if (video) {
    playbackRef.current = {
      time: video.currentTime,
      wasPlaying: !video.paused && !video.ended,
    };
  }

  setVideoMode(mode);
};


const [geminiCoaching, setGeminiCoaching] =
  useState<GeminiCoaching | null>(null);

const [coachingCached, setCoachingCached] =
    useState<boolean | null>(null);

const [isGeneratingCoaching, setIsGeneratingCoaching] =
  useState(false);

const [coachingError, setCoachingError] =
  useState<string | null>(null);


async function requestGeminiCoaching() {
  if (!analysis) return;

  setIsGeneratingCoaching(true);
  setCoachingError(null);

  try {
    const response = await fetch(
      `/api/coaching/${analysis.upload_id}`,
      {
        method: "POST"
      }
    );

    if (!response.ok) {
      throw new Error(
        `Gemini coaching request failed: ${response.status}`
      );
    }

    const data = await response.json();

    setGeminiCoaching(data.coaching);
    setCoachingCached(data.cached);
  } catch (error) {
    setCoachingError(
      error instanceof Error
        ? error.message
        : "Unable to generate Gemini coaching."
    );
  } finally {
    setIsGeneratingCoaching(false);
  }
}




  return <div className="min-h-screen">
    <header className="border-b border-[#263246] bg-[#101722]">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-5">
        <div className="flex items-center gap-3"><div className="rounded-xl bg-lime-400 p-2 text-slate-950"><Mountain size={23}/></div><div><div className="text-lg font-bold tracking-tight">AI Climbing Coach</div><div className="text-xs muted">Movement intelligence workspace</div></div></div>
        <div className="flex items-center gap-2 rounded-full border border-[#344257] px-3 py-2 text-xs"><span className={`h-2 w-2 rounded-full ${health === 'connected' ? 'bg-lime-400' : health === 'offline' ? 'bg-rose-400' : 'bg-amber-400'}`}/>{health === 'connected' ? 'API connected' : health === 'offline' ? 'API offline' : 'Connecting…'}</div>
      </div>
    </header>
    <main className="mx-auto max-w-7xl space-y-7 px-5 py-9">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <div className="mb-2 text-xs font-semibold uppercase tracking-[.25em] text-lime-400">Phase 06 · Video analysis</div>
          <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">Climb smarter. Review every move.</h1>
          <p className="mt-3 max-w-2xl muted">Review climbing footage, movement measurements, and evidence-aware coaching feedback in one place.</p>
        </div>
        <span className="rounded-full border border-lime-500/30 bg-lime-500/10 px-3 py-2 text-xs text-lime-300">{analysis ? 'Real movement metrics' : 'Awaiting analysis'}</span>
      </div>
      <VideoUploader
        onVideoSelected={selectVideo}
        onAnalysisComplete={(data) => {
          playbackRef.current = { time: 0, wasPlaying: false };
          setAnalysis(data);
          setVideoMode('skeleton');

          setGeminiCoaching(null);
          setCoachingError(null);
        }}
        onReset={() => {
          playbackRef.current = { time: 0, wasPlaying: false };
          setAnalysis(null);
          setVideoMode('original');

          setGeminiCoaching(null);
          setCoachingError(null);
        }}
      />
      <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
        <section className="panel overflow-hidden">
          <div className="flex items-center justify-between border-b border-[#263246] px-5 py-4">
            <h2 className="flex items-center gap-2 font-semibold"><Video size={19} className="text-lime-400"/>Climbing footage</h2>
            {analysis ? <div className="flex gap-2">
              <button type="button" onClick={() => switchVideoMode('original')} className={`rounded-lg px-3 py-1.5 text-xs ${videoMode === 'original' ? 'bg-lime-400 text-slate-950' : 'bg-slate-700 text-slate-200'}`}>Original</button>
              <button type="button" onClick={() => switchVideoMode('skeleton')} className={`rounded-lg px-3 py-1.5 text-xs ${videoMode === 'skeleton' ? 'bg-lime-400 text-slate-950' : 'bg-slate-700 text-slate-200'}`}>Skeleton overlay</button>
            </div> : <span className="text-xs muted">Local preview</span>}
          </div>
          <div className="flex aspect-video items-center justify-center bg-[#070b12]">
            {activeVideoUrl ? (
              <video
                ref={videoRef}
                key={activeVideoUrl}
                className="h-full w-full object-contain"
                controls
                playsInline
                src={activeVideoUrl}
                onLoadedMetadata={(event) => {
                  const video = event.currentTarget;

                  const savedTime = playbackRef.current.time;

                  if (Number.isFinite(video.duration)) {
                    video.currentTime = Math.min(
                      savedTime,
                      video.duration
                    );
                  } else {
                    video.currentTime = savedTime;
                  }

                  if (playbackRef.current.wasPlaying) {
                    video.play().catch(() => {
                      // Browser may block automatic playback.
                    });
                  }
                }}
              />
            ) : (
              <div className="text-center">
                <Clapperboard
                  className="mx-auto mb-3 text-slate-500"
                  size={46}
                />
                <p className="font-medium">No video selected</p>
                <p className="mt-1 text-sm muted">
                  Choose a climbing video in the uploader above.
                </p>
              </div>
            )}
          </div>
          <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-4"><div className="min-w-0 text-sm muted">{fileName || 'Select a video to preview it locally.'}</div><span className="text-xs muted">{analysis ? (videoMode === 'skeleton' ? 'MediaPipe skeleton overlay' : 'Original footage') : 'Preview before analysis'}</span></div>
        </section>
        <section className="panel p-5">
          <div className="mb-5 flex items-center justify-between"><h2 className="flex items-center gap-2 font-semibold"><Gauge size={19} className="text-lime-400"/>Movement metrics</h2><span className="text-xs muted">{analysis ? 'Uploaded video' : 'No analysis yet'}</span></div>
          <div className="space-y-3">{!analysis && <p className="rounded-xl border border-[#2b374a] bg-[#192232] p-4 text-sm text-slate-300">Select, upload, and analyze a video to see its movement metrics.</p>}{metrics.map(m => <div key={m.label} className="rounded-xl border border-[#2b374a] bg-[#192232] p-4"><div className="text-xs muted">{m.label}</div><div className="mt-1 text-2xl font-bold">{m.value}</div><div className="mt-1 text-xs text-slate-400">{m.note}</div></div>)}</div>
          <div className="mt-4 flex items-start gap-2 text-xs muted"><Info size={15} className="mt-0.5 shrink-0"/>{analysis ? 'Measurements from the uploaded video. Deep-bend durations depend on landmark visibility and classification thresholds.' : 'Previous analysis results are cleared whenever you select a new video.'}</div>
          {analysis && <a className="mt-4 inline-block text-sm text-lime-300 underline" href={analysis.metrics_url} target="_blank" rel="noreferrer">View full metrics JSON</a>}
        </section>
      </div>

      <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
        <section className="panel p-5">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="flex items-center gap-2 font-semibold">
              <BrainCircuit size={19} className="text-lime-400" />
              Coaching feedback
            </h2>

            <span className="text-xs muted">
              Evidence-based observations
            </span>
          </div>

          {!analysis ? (
            <p className="text-sm muted">
              Upload and analyze a climbing video to receive feedback.
            </p>
          ) : !analysis.feedback || analysis.feedback.length === 0 ? (
            <p className="text-sm muted">
              No qualifying movement observations were detected.
              This does not necessarily mean the climb was technically perfect.
            </p>
          ) : (
            <div className="space-y-4">
              {analysis.feedback.map((item, index) => (
                <div
                  key={`${item.body_part}-${item.start_seconds}-${index}`}
                  className="rounded-xl border border-[#334155] bg-[#192232] p-5"
                >

                  <div className="mb-3 flex items-center justify-between gap-3">
                    <h3 className="font-semibold capitalize text-lime-300">
                      {item.body_part.replaceAll("_", " ")}
                    </h3>
                    <button
                      type="button"
                      onClick={() => jumpToTimestamp(item.start_seconds)}
                      disabled={!activeVideoUrl}
                      className="rounded-lg border border-lime-500/30 bg-lime-500/10 px-3 py-1.5 text-xs font-medium text-lime-300 transition hover:bg-lime-500/20 disabled:cursor-not-allowed disabled:opacity-50"
                      title="Jump to this movement in the video"
                    >
                      ▶ Review {item.start_seconds.toFixed(2)}s –
                      {item.end_seconds.toFixed(2)}s
                    </button>
                  </div>
                  <div className="space-y-3 text-sm">
                    <div>
                      <p className="font-semibold text-white">
                        Observation
                      </p>
                      <p className="mt-1 text-slate-300">
                        {item.observation}
                      </p>
                    </div>

                    <div>
                      <p className="font-semibold text-white">
                        Possible interpretation
                      </p>
                      <p className="mt-1 text-slate-300">
                        {item.interpretation}
                      </p>
                    </div>

                    <div>
                      <p className="font-semibold text-white">
                        Coaching suggestion
                      </p>
                      <p className="mt-1 text-slate-300">
                        {item.recommendation}
                      </p>
                    </div>

                    <div>
                      <p className="font-semibold text-white">
                        Evidence limitations
                      </p>
                      <ul className="mt-1 list-disc space-y-1 pl-5 text-slate-400">
                        {item.limitations.map((limitation, i) => (
                          <li key={i}>{limitation}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
        <section className="panel p-5">
          <h2 className="flex items-center gap-2 font-semibold"><ShieldCheck size={19} className="text-lime-400"/>Pipeline status</h2>
          <div className="mt-5 space-y-4">{[
            ['Frontend interface', 'Ready'],
            ['FastAPI health endpoint', health === 'connected' ? 'Connected' : 'Not connected'],
            ['Video preview', videoUrl ? 'Ready' : 'Awaiting video'],
            ['Video upload endpoint', 'Available'],
            ['Analysis pipeline', analysis ? 'Completed' : 'Ready to run'],
            ['Gemini generation', isGeneratingCoaching ? 'Processing' : geminiCoaching ? 'Completed': 'Ready'],
          ].map(([name,status]) => <div key={name} className="flex items-center justify-between gap-3 text-sm"><span className="muted">{name}</span><span className="flex items-center gap-1.5 text-right text-slate-200">{['Ready','Connected','Completed','Available'].includes(status) ? <CheckCircle2 size={15} className="text-lime-400"/> : <ArrowUpRight size={15} className="text-slate-500"/>}{status}</span></div>)}</div>
          <div className="mt-6 rounded-xl bg-[#202b3a] p-4 text-xs leading-5 text-slate-300">Video uploads and movement analysis run through FastAPI. Skeleton-overlay video is converted to H.264 for browser playback. Gemini coaching can be generated on demand using the analyzed movement metricsf.</div>
        </section>
      </div>
    {analysis && (
      <section className="panel p-6">
      <div className="mb-5 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="flex items-center gap-2 text-xl font-semibold">
            <BrainCircuit size={22} className="text-lime-400" />
            Gemini AI Coaching
          </h2>

          <p className="mt-2 text-sm text-slate-400">
            AI-generated coaching recommendations based on
            your MediaPipe movement metrics.
          </p>
        </div>

        {/* Button and cache status */}
        <div className="flex flex-col items-end gap-2">
          <button
            type="button"
            onClick={requestGeminiCoaching}
            disabled={isGeneratingCoaching}
            className="rounded-lg bg-lime-400 px-5 py-2.5
                       font-semibold text-slate-950 transition
                       hover:bg-lime-300
                       disabled:cursor-not-allowed
                       disabled:opacity-50"
          >
            {isGeneratingCoaching
              ? "Generating Coaching..."
              : "Generate AI Coaching"}
          </button>

          {coachingCached !== null && geminiCoaching && (
            <p className="text-xs text-slate-400">
              {coachingCached
                ? "✓ Retrieved from cache — no Gemini API call needed"
                : "✓ Newly generated by Gemini and saved to cache"}
            </p>
          )}
        </div>
      </div>
            {coachingError && (
              <div className="mb-5 rounded-lg border border-red-500/30 bg-red-500/10 p-4">
                <p className="text-sm text-red-300">
                  {coachingError}
                </p>
              </div>
            )}

            {isGeneratingCoaching && (
              <div className="rounded-xl border border-[#334155] bg-[#192232] p-5">
                <p className="animate-pulse text-sm text-slate-300">
                  Gemini is analyzing your climbing metrics...
                </p>
              </div>
            )}

            {geminiCoaching && (
              <div className="space-y-6">

                {/* Overall Assessment */}
                <div className="rounded-xl border border-[#334155] bg-[#192232] p-5">
                  <h3 className="mb-3 text-lg font-semibold text-lime-300">
                    Overall Assessment
                  </h3>

                  <p className="text-sm leading-7 text-slate-300">
                    {geminiCoaching.overall_assessment}
                  </p>
                </div>

                {/* Strengths */}
                <div className="rounded-xl border border-[#334155] bg-[#192232] p-5">
                  <h3 className="mb-3 font-semibold text-white">
                    Strengths
                  </h3>

                  {geminiCoaching.strengths.length > 0 ? (
                    <ul className="list-disc space-y-2 pl-5 text-sm text-slate-300">
                      {geminiCoaching.strengths.map((strength, index) => (
                        <li key={index}>{strength}</li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-sm text-slate-400">
                      No specific strengths identified.
                    </p>
                  )}
                </div>

                {/* Improvements */}
                <div className="rounded-xl border border-[#334155] bg-[#192232] p-5">
                  <h3 className="mb-3 font-semibold text-white">
                    Areas to Explore
                  </h3>

                  {geminiCoaching.improvements.length > 0 ? (
                    <ul className="list-disc space-y-2 pl-5 text-sm text-slate-300">
                      {geminiCoaching.improvements.map((improvement, index) => (
                        <li key={index}>{improvement}</li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-sm text-slate-400">
                      No specific improvements suggested.
                    </p>
                  )}
                </div>

                {/* Limitations */}
                <div className="rounded-xl border border-[#334155] bg-[#192232] p-5">
                  <h3 className="mb-3 flex items-center gap-2 font-semibold text-white">
                    <Info size={17} className="text-amber-400" />
                    Analysis Limitations
                  </h3>

                  <ul className="list-disc space-y-2 pl-5 text-sm text-slate-400">
                    {geminiCoaching.limitations.map((limitation, index) => (
                      <li key={index}>{limitation}</li>
                    ))}
                  </ul>
                </div>

              </div>
            )}
          </section>
        )}
    </main>
  </div>;
}

import { useState } from 'react';
import './App.css';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const initialStatus = {
  message: 'Idle',
  tone: 'muted',
};

const tabs = [
  { id: 'design', label: 'Voice Design' },
  { id: 'clone', label: 'Voice Clone (Base)' },
  { id: 'custom', label: 'TTS (CustomVoice)' },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('design');
  const [designText, setDesignText] = useState(
    "It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!"
  );
  const [designLanguage, setDesignLanguage] = useState('auto');
  const [designVoice, setDesignVoice] = useState(
    'Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice.'
  );
  const [designAudioUrl, setDesignAudioUrl] = useState('');
  const [designStatus, setDesignStatus] = useState(initialStatus);
  const [designLoading, setDesignLoading] = useState(false);

  const [cloneText, setCloneText] = useState(
    'Hello! This is a short demo for the voice clone mode.'
  );
  const [cloneFile, setCloneFile] = useState(null);
  const [cloneAudioUrl, setCloneAudioUrl] = useState('');
  const [cloneStatus, setCloneStatus] = useState(initialStatus);
  const [cloneLoading, setCloneLoading] = useState(false);

  const [customText, setCustomText] = useState(
    'Welcome to Qwen3-TTS. This custom voice sample blends clarity with a calm, warm tone.'
  );
  const [customSpeaker, setCustomSpeaker] = useState('qwen3-en-female');
  const [customStyle, setCustomStyle] = useState('Warm, confident, and engaging.');
  const [customAudioUrl, setCustomAudioUrl] = useState('');
  const [customStatus, setCustomStatus] = useState(initialStatus);
  const [customLoading, setCustomLoading] = useState(false);

  const handleDesignSubmit = async (event) => {
    event.preventDefault();
    setDesignLoading(true);
    setDesignStatus({ message: 'Generating audio with the 12Hz 1.7B base model...', tone: 'info' });
    setDesignAudioUrl('');

    try {
      const response = await fetch(`${API_BASE_URL}/api/tts/design`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: designText,
          language: designLanguage,
          voice_description: designVoice,
        }),
      });

      if (!response.ok) {
        throw new Error('Generation failed');
      }

      const data = await response.json();
      setDesignAudioUrl(`${API_BASE_URL}${data.download_url}`);
      setDesignStatus({ message: 'Audio ready. Enjoy your preview!', tone: 'success' });
    } catch (error) {
      setDesignStatus({ message: error.message, tone: 'error' });
    } finally {
      setDesignLoading(false);
    }
  };

  const handleCloneSubmit = async (event) => {
    event.preventDefault();
    if (!cloneFile) {
      setCloneStatus({ message: 'Please add a reference audio file.', tone: 'error' });
      return;
    }

    setCloneLoading(true);
    setCloneStatus({ message: 'Cloning voice with the 12Hz 1.7B base model...', tone: 'info' });
    setCloneAudioUrl('');

    try {
      const formData = new FormData();
      formData.append('text', cloneText);
      formData.append('audio_file', cloneFile);

      const response = await fetch(`${API_BASE_URL}/api/tts/clone`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Voice clone failed');
      }

      const data = await response.json();
      setCloneAudioUrl(`${API_BASE_URL}${data.download_url}`);
      setCloneStatus({ message: 'Clone generated. You can listen now.', tone: 'success' });
    } catch (error) {
      setCloneStatus({ message: error.message, tone: 'error' });
    } finally {
      setCloneLoading(false);
    }
  };

  const handleCustomSubmit = async (event) => {
    event.preventDefault();
    setCustomLoading(true);
    setCustomStatus({ message: 'Rendering custom voice...', tone: 'info' });
    setCustomAudioUrl('');

    try {
      const response = await fetch(`${API_BASE_URL}/api/tts/custom`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: customText,
          speaker: customSpeaker,
          style: customStyle,
        }),
      });

      if (!response.ok) {
        throw new Error('Custom voice generation failed');
      }

      const data = await response.json();
      setCustomAudioUrl(`${API_BASE_URL}${data.download_url}`);
      setCustomStatus({ message: 'Custom voice ready for playback.', tone: 'success' });
    } catch (error) {
      setCustomStatus({ message: error.message, tone: 'error' });
    } finally {
      setCustomLoading(false);
    }
  };

  return (
    <div className="page">
      <div className="header">
        <h1>Qwen3-TTS Demo</h1>
        <p>
          A unified Text-to-Speech demo featuring three powerful modes:
        </p>
        <ul>
          <li>
            <strong>Voice Design</strong>: Create custom voices using natural language descriptions
          </li>
          <li>
            <strong>Voice Clone (Base)</strong>: Clone any voice from a reference audio
          </li>
          <li>
            <strong>TTS (CustomVoice)</strong>: Generate speech with predefined speakers and optional
            style instructions. Built with{' '}
            <a href="https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base" target="_blank" rel="noreferrer">
              Qwen3-TTS-12Hz-1.7B-Base
            </a>{' '}
            by Alibaba Qwen Team.
          </li>
        </ul>
      </div>

      <div className="tabs">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            className={activeTab === tab.id ? 'tab active' : 'tab'}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === 'design' && (
        <div className="tab-panel">
          <div className="panel-body">
            <section className="panel">
              <div className="panel-title">Text to Synthesize</div>
              <form onSubmit={handleDesignSubmit} className="panel-form">
                <textarea
                  value={designText}
                  onChange={(event) => setDesignText(event.target.value)}
                />

                <div className="field">
                  <label>Language</label>
                  <select
                    value={designLanguage}
                    onChange={(event) => setDesignLanguage(event.target.value)}
                  >
                    <option value="auto">Auto</option>
                    <option value="en">English</option>
                    <option value="de">German</option>
                    <option value="zh">Chinese</option>
                  </select>
                </div>

                <div className="field">
                  <label>Voice Description</label>
                  <textarea
                    value={designVoice}
                    onChange={(event) => setDesignVoice(event.target.value)}
                  />
                </div>

                <button className="primary" type="submit" disabled={designLoading}>
                  {designLoading ? 'Generating...' : 'Generate with Custom Voice'}
                </button>
              </form>
            </section>

            <aside className="side-panel">
              <div className="panel">
                <div className="panel-title">Generated Audio</div>
                <div className="audio-box">
                  {designAudioUrl ? (
                    <audio controls src={designAudioUrl} />
                  ) : (
                    <span className="audio-placeholder">♪</span>
                  )}
                </div>
              </div>
              <div className="panel">
                <div className="panel-title">Status</div>
                <p className={`status ${designStatus.tone}`}>{designStatus.message}</p>
              </div>
            </aside>
          </div>
        </div>
      )}

      {activeTab === 'clone' && (
        <div className="tab-panel">
          <div className="panel-body">
            <section className="panel">
              <div className="panel-title">Voice Clone (Base)</div>
              <form onSubmit={handleCloneSubmit} className="panel-form">
                <textarea
                  value={cloneText}
                  onChange={(event) => setCloneText(event.target.value)}
                />

                <div className="field">
                  <label>Reference Audio</label>
                  <input
                    type="file"
                    accept="audio/*"
                    onChange={(event) => setCloneFile(event.target.files?.[0] ?? null)}
                  />
                </div>

                <button className="primary" type="submit" disabled={cloneLoading}>
                  {cloneLoading ? 'Cloning...' : 'Generate Clone'}
                </button>
              </form>
            </section>

            <aside className="side-panel">
              <div className="panel">
                <div className="panel-title">Generated Audio</div>
                <div className="audio-box">
                  {cloneAudioUrl ? (
                    <audio controls src={cloneAudioUrl} />
                  ) : (
                    <span className="audio-placeholder">♪</span>
                  )}
                </div>
              </div>
              <div className="panel">
                <div className="panel-title">Status</div>
                <p className={`status ${cloneStatus.tone}`}>{cloneStatus.message}</p>
              </div>
            </aside>
          </div>
        </div>
      )}

      {activeTab === 'custom' && (
        <div className="tab-panel">
          <div className="panel-body">
            <section className="panel">
              <div className="panel-title">TTS (CustomVoice)</div>
              <form onSubmit={handleCustomSubmit} className="panel-form">
                <textarea
                  value={customText}
                  onChange={(event) => setCustomText(event.target.value)}
                />

                <div className="field">
                  <label>Speaker</label>
                  <select
                    value={customSpeaker}
                    onChange={(event) => setCustomSpeaker(event.target.value)}
                  >
                    <option value="qwen3-en-female">Qwen3 English Female</option>
                    <option value="qwen3-en-male">Qwen3 English Male</option>
                    <option value="qwen3-de-female">Qwen3 German Female</option>
                    <option value="qwen3-zh-female">Qwen3 Chinese Female</option>
                  </select>
                </div>

                <div className="field">
                  <label>Style Instructions</label>
                  <textarea
                    value={customStyle}
                    onChange={(event) => setCustomStyle(event.target.value)}
                  />
                </div>

                <button className="primary" type="submit" disabled={customLoading}>
                  {customLoading ? 'Generating...' : 'Generate Custom Voice'}
                </button>
              </form>
            </section>

            <aside className="side-panel">
              <div className="panel">
                <div className="panel-title">Generated Audio</div>
                <div className="audio-box">
                  {customAudioUrl ? (
                    <audio controls src={customAudioUrl} />
                  ) : (
                    <span className="audio-placeholder">♪</span>
                  )}
                </div>
              </div>
              <div className="panel">
                <div className="panel-title">Status</div>
                <p className={`status ${customStatus.tone}`}>{customStatus.message}</p>
              </div>
            </aside>
          </div>
        </div>
      )}
    </div>
  );
}

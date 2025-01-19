import React, { useState, useEffect } from "react";
import axios from "axios";
import "./speech_text.css";

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const mic = new SpeechRecognition();

mic.continuous = true;
mic.interimResults = true;
mic.lang = "en-US";

function SpeechText2() {
  const [isListening, setIsListening] = useState(false);
  const [note, setNote] = useState("");
  const [savedNotes, setSavedNotes] = useState([]);
  const [query, setQuery] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);

  useEffect(() => {
    const handleMic = () => {
      if (isListening) {
        mic.start();
        mic.onend = mic.start;
      } else {
        mic.stop();
        mic.onend = () => console.log("Stopped Mic on Click");
      }
    };

    mic.onresult = (event) => {
      const transcript = Array.from(event.results)
        .map((result) => result[0].transcript)
        .join("");
      setNote(transcript);
    };

    mic.onerror = (event) => console.log(event.error);

    handleMic();

    return () => {
      mic.stop();
      mic.onend = null;
      mic.onresult = null;
      mic.onerror = null;
    };
  }, [isListening]);

  const handleSaveNote = () => {
    setSavedNotes([...savedNotes, { text: note, time: new Date().toLocaleString() }]);
    setNote("");
  };

  useEffect(() => {
    if (savedNotes.length > 0 && !query) {
      axios.post("http://192.168.14.240:8000/create_vector_store", { input: savedNotes.map(note => note.text) })
        .then(response => console.log(response))
        .catch(error => console.log(error));
    }
  }, [savedNotes]);

  useEffect(() => {
    if (query && savedNotes.length > 0) {
      axios.post("http://192.168.14.240:8000/query_audio", { input: savedNotes.map(note => note.text) }, { responseType: "arraybuffer" })
        .then(response => setAudioUrl(URL.createObjectURL(new Blob([response.data], { type: "audio/mp3" }))))
        .catch(error => console.log(error));
    }
  }, [query, savedNotes]);

  const handleQuery = () => {
    setIsListening(true);
    setQuery(true);
  };

  const toggleListening = () => setIsListening(prev => !prev);

  return (
    <main>
      <div className="voice-recorder">
        <div className="current-note">
          <h2>{isListening ? "Click to stop recording" : "Click to start recording"}</h2>
          <div className="mic-status">
            {isListening ? <span className="mic-on">🎙️ (Recording)</span> : <span className="mic-off">🛑🎙️</span>}
          </div>
          <div className="actions">
            <button onClick={toggleListening}>{isListening ? "Stop" : "Start"}</button>
            <button onClick={handleSaveNote} disabled={!note || query}>Save Note</button>
          </div>
          <p>{note}</p>
        </div>
        <div className="saved-notes">
          <div style={{marginBottom: '1rem'}}>
          <button onClick={handleQuery} disabled={savedNotes.length === 0}>Ask Question</button>
          </div>
          <div>
          {audioUrl && <audio controls src={audioUrl} onPlay={() => setQuery(false)}></audio>}
          </div>
        </div>
      </div>
    </main>
  );
}

export default SpeechText2;


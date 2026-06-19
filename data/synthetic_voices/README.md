# Synthetic voices (ElevenLabs)

The synthetic half of the study: **30 ElevenLabs voices** — 6 per accent group (3 male, 3 female) across American,
British, Chinese, Indian, and Nigerian English. Each voice reads scripts in multiple **domains**, so the dataset has a
`domain` column:

| Domain | What it is | Clips |
|---|---|---|
| `hiring` | the 8 hiring-interview scripts (4 categories × fluent/disfluent) | 240 |
| `presentation` | academic-presentation scripts (education): 4 subjects × {conceptual, technical, presentation} × fluent/disfluent | 720 |

…for **960 clips** total. (`english-test` / immigration is coming next.) Because the speakers are commercial TTS voices,
not people, no anonymization is needed.

**Files here**
| File | What it is |
|---|---|
| `speaker_ids.csv` | The 30 voices: `name, accent, language, gender, voice_id` (+ ElevenLabs voice notes). |
| `scripts.csv` | The **hiring** scripts (`category, filename, script`). |
| `elevenlabs_metadata.csv` | One row per clip — `file_name, speaker_name, gender, voice_id, accent, language, script_filename, script_text, category, script_type, domain`. Mirrors the Hugging Face dataset's `metadata.csv`. |

**How the audio + metadata were made.** `models/run_elevenlabs.py` synthesizes every (voice × script) pair via the
ElevenLabs API (the hiring set is driven by `speaker_ids.csv` + `scripts.csv`; the presentation set uses the academic
scripts, whose text is preserved per row in `elevenlabs_metadata.csv`):

```bash
python models/run_elevenlabs.py    # reads speaker_ids.csv + scripts.csv, writes audio_samples/elevenlabs/
```

**Audio is not stored in this repo.** It lives in the public Hugging Face dataset
[`multispeak/accent-synthetic-voices`](https://huggingface.co/datasets/multispeak/accent-synthetic-voices)
(`load_dataset(...)`, filter by `domain`); download it into `audio_samples/elevenlabs/` to re-run the synthetic
evaluations, or regenerate it with the command above (needs `ELEVENLABS_API_KEY`).

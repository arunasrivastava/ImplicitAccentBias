# Synthetic voices (ElevenLabs)

The synthetic half of the study: **30 ElevenLabs voices** — 6 per accent group (3 male, 3 female) across
American, British, Chinese, Indian, and Nigerian English — each reading the **8 hiring scripts** (4 categories
× fluent/disfluent variants), for **240 clips**. Because the speakers are commercial TTS voices, not people, no
anonymization is needed.

**Files here**
| File | What it is |
|---|---|
| `speaker_ids.csv` | The 30 voices: `name, accent, language, gender, voice_id` (+ ElevenLabs voice notes). |
| `scripts.csv` | The scripts each voice reads: `category, filename, script`. |
| `elevenlabs_metadata.csv` | One row per generated clip (voice × script): the manifest that `src/run_evaluation.py` reads in synthetic mode. |

**How the audio + metadata were made.** `models/run_elevenlabs.py` synthesizes every (voice × script) pair via the
ElevenLabs API and writes the per-speaker WAVs plus `elevenlabs_metadata.csv`:

```bash
python models/run_elevenlabs.py    # reads speaker_ids.csv + scripts.csv, writes audio_samples/elevenlabs/
```

**Audio is not stored in this repo.** The generated WAVs live in the
[multispeak Hugging Face organization](https://huggingface.co/multispeak); download them into
`audio_samples/elevenlabs/` to re-run the synthetic evaluations, or regenerate them with the command above
(needs `ELEVENLABS_API_KEY`).

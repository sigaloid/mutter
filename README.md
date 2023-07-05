# Mutter

Mutter is a Rust library that makes transcription with the OpenAI Whisper models, easy.

```rust
use mutter::{Model, ModelType};

let model = Model::download(&ModelType::BaseEn).unwrap();
let mp3: Vec<u8> = download_mp3();
let transcription = model.transcribe_audio(mp3, false, false, None).unwrap();
println!("{}", transcription.as_text());
println!("{}", transcription.as_srt());
```

# Codecs

Mutter supports all codecs that Rodio, the audio backend, supports.
* MP3 (Symphonia)
* WAV (Hound)
* OGG Vorbis (lewton)
* FLAC (claxon)

Alternatively, enable the `minimp3` feature to use the minimp3 backend.

You can also enable any of these features to enable the optional symphonia backend for these features.


```toml
symphonia-aac = ["rodio/symphonia-aac"]
symphonia-all = ["rodio/symphonia-all"]
symphonia-flac = ["rodio/symphonia-flac"]
symphonia-isomp4 = ["rodio/symphonia-isomp4"]
symphonia-mp3 = ["rodio/symphonia-mp3"]
symphonia-vorbis = ["rodio/symphonia-vorbis"]
symphonia-wav = ["rodio/symphonia-wav"]
```

# About this crate

This crate is largely a thin wrapper around whisper-rs, that simply opens up transcription to any file format (it handles conversion via `rodio`). Whisper-rs handles the actual bindings to the Whisper.cpp library. I wrote this because I didn't want to reimplement the conversion + re-encoding to 16-bit mono PCM WAV every single time I wanted to use Whisper in a new Rust library, and my initial implementation relying on `ffmpeg` existing on the target device was not at all compatible. While the targets are limited by whisper-rs and by extension whisper.cpp's supported targets, and while ffmpeg is pretty universal, I wanted to portable-ify as much as possible. In addition, my university's compute clusters don't have FFmpeg by default.

Oh, and I added a download function that will load the model at runtime. Be warned, the models can be as large as 3GB!

# Transcoding

This crate relies on rodio to perform the transcoding. In order to reduce background noise and optimize for human speech, it also applies a 200hz low pass filter and a 3000hz high pass filter. I wanted to apply more advanced voice filters, like FFmpeg's `arnndn`, but was unable to do so while keeping it within the Rodio ecosystem for simplicity.

# Future work

I would love to extend this crate to have more advanced noise reduction. Outside of that, I'd love to explore any more opinionated modifications to the audio, like implementing a VAD, but I'm yet to find any crates in the ecosystem yet (and might not have the time to implement it if I did). 

# Credits

@tazz4843 for their wonderful work on the [whisper-rs](https://github.com/tazz4843/whisper-rs) bindings. This crate essentially adds just two features above it: transcoding, and downloading models.

### Show appreciation

Want to say thanks for this library? Just click the button below and leave a brief note. It would make my day :)

[![Click me to show appreciation](https://img.shields.io/badge/Say%20Thanks-%F0%9F%A6%80%F0%9F%A6%80%F0%9F%A6%80-1EAEDB.svg)](https://saythanks.io/to/sigaloid)
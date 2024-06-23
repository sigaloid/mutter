//! # Mutter
//!
//! Mutter is a Rust library that makes transcription with the Whisper transcription models
//! easy and accesible in Rust. It's a wrapper around [whisper-rs](https://github.com/tazz4843/whisper-rs) which is in turn
//! a wrapper around [whisper.cpp](https://github.com/ggerganov/whisper.cpp).
//!
//! ```no_run
//! use mutter::{Model, ModelType};
//!
//! let model = Model::download(&ModelType::BaseEn).unwrap();
//! let mp3: Vec<u8> = download_mp3(); // Your own function to download audio
//! let translate = false;
//! let individual_word_timestamps = false;
//! let threads = Some(8);
//! let transcription = model.transcribe_audio(mp3, translate, individual_word_timestamps, threads).unwrap();
//! println!("{}", transcription.as_text());
//! println!("{}", transcription.as_srt());
//! ```
//!
//! # Codecs
//!
//! Mutter supports all codecs that Rodio, the audio backend, supports.
//! * MP3 (Symphonia)
//! * WAV (Hound)
//! * OGG Vorbis (lewton)
//! * FLAC (claxon)
//!
//! Alternatively, enable the `minimp3` feature to use the minimp3 backend.
//!
//! You can also enable any of these features to enable the optional symphonia backend for these features.
//!
//!
//! ```toml
//! symphonia-aac = ["rodio/symphonia-aac"]
//! symphonia-all = ["rodio/symphonia-all"]
//! symphonia-flac = ["rodio/symphonia-flac"]
//! symphonia-isomp4 = ["rodio/symphonia-isomp4"]
//! symphonia-mp3 = ["rodio/symphonia-mp3"]
//! symphonia-vorbis = ["rodio/symphonia-vorbis"]
//! symphonia-wav = ["rodio/symphonia-wav"]
//! ```
//!
#![deny(
    anonymous_parameters,
    clippy::all,
    late_bound_lifetime_arguments,
    path_statements,
    patterns_in_fns_without_body,
    trivial_numeric_casts,
    unused_extern_crates
)]
#![warn(
    clippy::dbg_macro,
    clippy::decimal_literal_representation,
    clippy::get_unwrap,
    clippy::nursery,
    clippy::pedantic,
    clippy::todo,
    clippy::unimplemented,
    clippy::use_debug,
    clippy::all,
    unused_qualifications,
    variant_size_differences
)]
#![allow(clippy::must_use_candidate)]
use std::{fmt::Display, time::Instant};

use log::{info, trace};
use strum::EnumIter;
use transcript::{Transcript, Utterance};
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperError,
};

mod tests;
mod transcode;
pub mod transcript;

/// Model struct. Can be constructed with [`Model::new`] or [`Model::download`].
/// Contains the Whisper model and its context.
pub struct Model {
    context: WhisperContext,
}

impl Model {
    /// Creates a new model from a model path. Must be a path to a valid Whisper model,
    /// in GGML format, that is compatible with Whisper.cpp.
    /// # Arguments
    /// - `path`: Path to the model.
    /// # Errors
    /// - [`WhisperError`]
    pub fn new(path: &str) -> Result<Self, WhisperError> {
        trace!("Loading model {}", path);
        // Sanity check - make sure the path exists
        let path_converted = std::path::Path::new(path);
        if !path_converted.exists() {
            return Err(WhisperError::InitError);
        }

        let params: WhisperContextParameters = WhisperContextParameters::default();
        Ok({
            Self {
                context: WhisperContext::new_with_params(path, params)?,
            }
        })
    }

    /// Creates a new model and downloads the specified model type from huggingface.
    /// # Arguments
    /// - `model`: [`ModelType`].
    /// # Errors
    /// - [`ModelError`]
    ///     - [`ModelError::WhisperError`],
    ///     - [`ModelError::DownloadError`],
    ///     - [`ModelError::IoError`],
    /// # Panics
    /// This function shouldn't panic, but may due to the underlying -sys bindings.
    /// It shouldn't panic within _this_ crate.

    pub fn download(model: &ModelType) -> Result<Self, ModelError> {
        trace!("Downloading model {}", model);
        let resp = ureq::get(&model.to_string())
            .call()
            .map_err(|e| ModelError::DownloadError(Box::new(e)))?;
        assert!(resp.has("Content-Length"));
        let len: usize = resp
            .header("Content-Length")
            .unwrap()
            .parse()
            .unwrap_or_default();
        trace!("Model length: {}", len);
        let mut bytes: Vec<u8> = Vec::with_capacity(len);
        resp.into_reader()
            .read_to_end(&mut bytes)
            .map_err(ModelError::IoError)?;
        assert_eq!(bytes.len(), len);
        info!("Downloaded model: {}", model);
        let params: WhisperContextParameters = WhisperContextParameters::default();

        Ok({
            Self {
                context: WhisperContext::new_from_buffer_with_params(&bytes, params)
                    .map_err(ModelError::WhisperError)?,
            }
        })
    }

    /// Transcribes audio to text, given the audio is a byte array of a file.
    /// Supported codecs: MP3 (Symphonia), WAV (Hound), OGG Vorbis (lewton),
    /// FLAC (claxon).
    ///
    /// # Arguments
    /// - `audio`: Audio to transcribe. An array of bytes.
    /// - `translate`: Whether to translate the text.
    /// - `word_timestamps`: Whether to output word timestamps.
    /// - `initial_prompt`: Optinal initial prompt to whisper model.
    /// - `language`: Optinal language setting for whisper model.
    /// - `threads`: Number of threads to use. `None` will use the number of cores from
    /// the `num_cpus` crate.
    /// # Errors
    /// - [`ModelError`]
    /// # Returns
    /// [Transcript]    
    pub fn transcribe_audio(
        &self,
        audio: impl AsRef<[u8]>,
        translate: bool,
        word_timestamps: bool,
        initial_prompt: Option<&str>,
        language: Option<&str>,
        threads: Option<u16>,
    ) -> Result<Transcript, ModelError> {
        trace!("Decoding audio.");
        let samples = transcode::decode(audio.as_ref().to_vec())?;
        trace!("Transcribing audio.");
        self.transcribe_pcm_s16le(
            &samples,
            translate,
            word_timestamps,
            initial_prompt,
            language,
            threads,
        )
    }

    /// Transcribes audio to text, given the audio is an [f32] float array of codec
    /// `pcm_s16le` and in single-channel format.
    ///
    /// You probably want to use [`Model::transcribe_audio`] instead, unless you've already
    /// converted it into the correct format.
    ///
    /// # Arguments
    /// - `audio`: Audio to transcribe. Must be a [f32] array.
    /// - `translate`: Whether to translate the text.
    /// - `word_timestamps`: Whether to output word timestamps.
    /// - `initial_prompt`: Optinal initial prompt to whisper model.
    /// - `language`: Optinal language setting for whisper model.
    /// - `threads`: Number of threads to use. `None` will use the number of cores from
    ///
    /// # Errors
    /// - [`ModelError`]
    /// # Panics
    /// This function shouldn't panic, but may due to the underlying -sys c bindings.
    /// # Returns
    /// [Transcript]
    pub fn transcribe_pcm_s16le(
        &self,
        audio: &[f32],
        translate: bool,
        word_timestamps: bool,
        initial_prompt: Option<&str>,
        language: Option<&str>,
        threads: Option<u16>,
    ) -> Result<Transcript, ModelError> {
        trace!(
            "Transcribing audio: {} with translate: {translate} and timestamps: {word_timestamps}",
            audio.len()
        );

        let mut params = FullParams::new(SamplingStrategy::BeamSearch {
            beam_size: 5,
            patience: 1.0,
        });

        if let Some(prompt) = initial_prompt {
            params.set_initial_prompt(prompt);
        }

        params.set_language(language);

        params.set_translate(translate);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_token_timestamps(word_timestamps);
        params.set_split_on_word(true);

        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
        let threads = threads.map_or_else(|| num_cpus::get() as i32, i32::from);

        trace!("Using {} threads", threads);

        params.set_n_threads(threads);

        let st = Instant::now();
        let mut state = self.context.create_state().expect("failed to create state");
        trace!("Transcribing audio with WhisperState");
        state.full(params, audio).expect("failed to transcribe");

        let num_segments = state.full_n_segments().expect("failed to get segments");
        trace!("Number of segments: {}", num_segments);

        let mut words = Vec::new();
        let mut utterances = Vec::new();
        for segment_idx in 0..num_segments {
            let text = state
                .full_get_segment_text(segment_idx)
                .map_err(ModelError::WhisperError)?;
            let start = state
                .full_get_segment_t0(segment_idx)
                .map_err(ModelError::WhisperError)?;
            let stop = state
                .full_get_segment_t1(segment_idx)
                .map_err(ModelError::WhisperError)?;

            utterances.push(Utterance { start, stop, text });

            if !word_timestamps {
                trace!("Skipping word timestamps");
                continue;
            }

            trace!("Getting word timestamps for segment {}", segment_idx);

            let num_tokens = state
                .full_n_tokens(segment_idx)
                .map_err(ModelError::WhisperError)?;

            for t in 0..num_tokens {
                let text = state
                    .full_get_token_text(segment_idx, t)
                    .map_err(ModelError::WhisperError)?;
                let token_data = state
                    .full_get_token_data(segment_idx, t)
                    .map_err(ModelError::WhisperError)?;

                if text.starts_with("[_") {
                    continue;
                }

                words.push(Utterance {
                    text,
                    start: token_data.t0,
                    stop: token_data.t1,
                });
            }
        }

        Ok(Transcript {
            utterances,
            processing_time: Instant::now().duration_since(st),
            word_utterances: if word_timestamps { Some(words) } else { None },
        })
    }
}
/// Crate error that contains an enum of all possible errors related to the model.
#[derive(Debug)]
pub enum ModelError {
    /// [`WhisperError`]. Error either loading model, or during transcription, in the
    /// actual whisper.cpp library
    WhisperError(WhisperError),
    /// [`ureq::Error`]. Error downloading model.
    DownloadError(Box<ureq::Error>),
    /// [`std::io::Error`]. Error reading model.
    IoError(std::io::Error),
    /// [`AudioDecodeError`]. Error decoding audio.
    AudioDecodeError,
}

#[derive(Debug, EnumIter)]
pub enum ModelType {
    /// Tiny Whisper model - finetuned for English.
    /// Size: 75 MB.
    TinyEn,
    /// Tiny Whisper model.
    /// Size: 75 MB.
    Tiny,

    /// Base Whisper model - finetuned for English.
    /// Size: 142 MB.
    BaseEn,

    /// Base Whisper model.
    /// Size: 142 MB.
    Base,

    /// Small Whisper model - finetuned for English.
    /// Size: 466 MB.
    SmallEn,

    /// Small Whisper model.
    /// Size: 466 MB.
    Small,

    /// Medium Whisper model - finetuned for English.
    /// Size: 1.5 GB.
    MediumEn,

    /// Medium Whisper model.
    /// Size: 1.5 GB.
    Medium,

    /// Large Whisper model - old version.
    /// Size: 2.9 GB.
    LargeV1,

    /// Large Whisper model - V2.
    /// Size: 2.9 GB.
    LargeV2,

    /// Large Whisper model - V3.
    /// Size: 2.9 GB.
    LargeV3,
}

impl Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-"
        )?;
        match self {
            Self::TinyEn => write!(f, "tiny.en.bin"),
            Self::Tiny => write!(f, "tiny.bin"),
            Self::BaseEn => write!(f, "base.en.bin"),
            Self::Base => write!(f, "base.bin"),
            Self::SmallEn => write!(f, "small.en.bin"),
            Self::Small => write!(f, "small.bin"),
            Self::MediumEn => write!(f, "medium.en.bin"),
            Self::Medium => write!(f, "medium.bin"),
            Self::LargeV1 => write!(f, "large-v1.bin"),
            Self::LargeV2 => write!(f, "large-v2.bin"),
            Self::LargeV3 => write!(f, "large-v3.bin"),
        }
    }
}

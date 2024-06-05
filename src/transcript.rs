use num::integer::div_floor;
use serde::{Deserialize, Serialize};
use std::time::Duration;

pub use crate::Model;

// Repurposed from https://github.com/m1guelpf/whisper-cli-rs/. Fixed numerous bugs/typos

/// Transcript of an audio.
#[derive(Debug, Serialize, Deserialize)]
pub struct Transcript {
    /// Duration that it took to transcribe the audio.
    pub processing_time: Duration,
    /// List of utterances in the transcript - split by normal segments.
    pub utterances: Vec<Utterance>,
    /// List of words in the transcript - split by each word.
    /// Only present if `word_timestamps` is `true` in [`Model::transcribe_audio`].
    pub word_utterances: Option<Vec<Utterance>>,
}

/// A single utterance in the transcript.
/// Contains a start and stop timestamp.
/// Also contains the text of the utterance.
#[derive(Debug, Serialize, Deserialize)]
pub struct Utterance {
    /// Timestamp of the start of the utterance, in ms
    pub start: i64,
    /// Timestamp of the end of the utterance, in ms
    pub stop: i64,
    /// Text of the utterance.
    pub text: String,
}

impl Transcript {
    /// Returns the transcript as a string.
    #[must_use]
    pub fn as_text(&self) -> String {
        self.utterances
            .iter()
            .fold(String::new(), |transcript, fragment| {
                transcript + format!("{}\n", fragment.text.trim()).as_str()
            })
    }

    /// Returns the transcript in VTT format.
    #[must_use]
    pub fn as_vtt(&self) -> String {
        let vtt = self
            .utterances
            .iter()
            .fold(String::new(), |transcript, fragment| {
                transcript
                    + format!(
                        "{} --> {}\n{}\n\n",
                        format_timestamp(fragment.start, true, "."),
                        format_timestamp(fragment.stop, true, "."),
                        fragment.text.trim().replace("-->", "->")
                    )
                    .as_str()
            });
        format!("WEBVTT\n{vtt}")
    }

    /// Returns the transcript in SRT format.
    #[must_use]
    pub fn as_srt(&self) -> String {
        self.utterances
            .iter()
            .fold((1, String::new()), |(i, transcript), fragment| {
                (
                    i + 1,
                    transcript
                        + format!(
                            "{i}\n{} --> {}\n{}\n",
                            format_timestamp(fragment.start, true, ","),
                            format_timestamp(fragment.stop, true, ","),
                            fragment.text.trim().replace("-->", "->")
                        )
                        .as_str(),
                )
            })
            .1
    }
}

/// Timestamp is oddly given in number of seconds * 100, or number of milliseconds / 10.
/// This function corrects it and formats it in the desired format.
fn format_timestamp(num: i64, always_include_hours: bool, decimal_marker: &str) -> String {
    assert!(num >= 0, "non-negative timestamp expected");
    let mut milliseconds: i64 = num * 10;

    let hours = div_floor(milliseconds, 3_600_000);
    milliseconds -= hours * 3_600_000;

    let minutes = div_floor(milliseconds, 60_000);
    milliseconds -= minutes * 60_000;

    let seconds = div_floor(milliseconds, 1_000);
    milliseconds -= seconds * 1_000;

    let hours_marker = if always_include_hours || hours != 0 {
        format!("{hours:02}:")
    } else {
        String::new()
    };

    format!("{hours_marker}{minutes:02}:{seconds:02}{decimal_marker}{milliseconds:03}")
}

#[test]
fn test_format_timestamp() {
    let result = format_timestamp(100, true, ".");
    assert_eq!(result, "00:00:01.000");
}

#[test]
fn test_format_timestamp_hours() {
    let result = format_timestamp(100, true, ",");
    assert_eq!(result, "00:00:01,000");
}

#[test]
fn test_format_timestamp_seconds() {
    let result = format_timestamp(100, false, ".");
    assert_eq!(result, "00:01.000");
}

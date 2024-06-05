use rodio::{source::UniformSourceIterator, Decoder, Source};
use std::io::Cursor;

use crate::ModelError;

/// Decode a byte array of audio into a float array
pub fn decode(bytes: Vec<u8>) -> Result<Vec<f32>, ModelError> {
    let input = Cursor::new(bytes);
    let source = Decoder::new(input).unwrap();
    let output_sample_rate = 16000;
    let channels = 1;
    // Resample to output sample rate and channels
    let resample = UniformSourceIterator::new(source, channels, output_sample_rate);
    // High and low pass filters to enhance the audio
    let pass_filter = resample.low_pass(3000).high_pass(200).convert_samples();
    let samples: Vec<i16> = pass_filter.collect::<Vec<i16>>();
    let mut output: Vec<f32> = vec![0.0f32; samples.len()];
    let result: Result<(), whisper_rs::WhisperError> =
        whisper_rs::convert_integer_to_float_audio(&samples, &mut output);
    result.map(|()| output).map_err(ModelError::WhisperError)
}

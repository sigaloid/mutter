use rodio::{source::UniformSourceIterator, Decoder, Source};
use std::io::Cursor;

pub fn decode(bytes: Vec<u8>) -> Vec<f32> {
    let input = Cursor::new(bytes);
    let source = Decoder::new(input).unwrap();
    let output_sample_rate = 16000;
    let channels = 1;
    // Resample to output sample rate and channels
    let resample = UniformSourceIterator::new(source, channels, output_sample_rate);
    // High and low pass filters to enhance the audio
    let pass_filter = resample.low_pass(3000).high_pass(200).convert_samples();

    whisper_rs::convert_integer_to_float_audio(&pass_filter.collect::<Vec<i16>>())
}

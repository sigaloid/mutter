// ModelType tests
#[cfg(test)]
use {
    crate::{Model, ModelType},
    audrey::hound::WavReader,
    std::io::Cursor,
    strum::IntoEnumIterator,
};

#[test]
fn test_model_urls() {
    for model in ModelType::iter() {
        let url = model.to_string();
        println!("Testing model: {url}");
        let head = ureq::head(&url)
            .call()
            .expect("Failed to send Head request");
        assert!(head.status() == 200);
        assert!(head.header("Content-Length").is_some());
        let len: usize = head
            .header("Content-Length")
            .unwrap()
            .parse()
            .unwrap_or_default();
        // Larger than the smallest model. Basiclally just check huggingface has resolved
        // the download URL correctly
        assert!(len > 77_600_000);
    }
}

#[test]
fn test_transcribe() {
    let model = Model::download(&ModelType::TinyEn).unwrap();
    let jfk_wav = include_bytes!("../samples/jfk.wav");

    let mut reader = WavReader::new(Cursor::new(jfk_wav)).unwrap();
    let samples: Result<Vec<i16>, _> = reader.samples().collect();
    let audio = whisper_rs::convert_integer_to_float_audio(&samples.unwrap());

    let transcription = model
        .transcribe_pcm_s16le(&audio, false, false, None)
        .unwrap();
    assert!(transcription.as_text().contains("country"));
}

#[test]
fn test_transcribe_with_transcode() {
    let model = Model::download(&ModelType::TinyEn).unwrap();
    let kliks_mp3 = include_bytes!("../samples/3kliks-cut.mp3");

    let transcription = model
        .transcribe_audio(kliks_mp3, false, false, None)
        .unwrap();
    assert!(transcription.as_text().contains("Valve"));
}

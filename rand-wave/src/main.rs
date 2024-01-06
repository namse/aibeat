use anyhow::*;

fn main() -> Result<()> {
    let mut cymbals = open_wavs("cymbals")?;
    let mut kicks = open_wavs("kicks")?;
    let mut snares = open_wavs("snares")?;

    let kick_index: usize = rand::random::<usize>() % kicks.len();
    let cymbals_indexes: [usize; 4] = [
        rand::random::<usize>() % cymbals.len(),
        rand::random::<usize>() % cymbals.len(),
        rand::random::<usize>() % cymbals.len(),
        rand::random::<usize>() % cymbals.len(),
    ];
    let snare_index: usize = rand::random::<usize>() % snares.len();

    let kick = get_samples(&mut kicks[kick_index])?;
    let cymbals = cymbals_indexes
        .iter()
        .map(|i| get_samples(&mut cymbals[*i]))
        .collect::<Result<Vec<_>>>()?;

    let snare = get_samples(&mut snares[snare_index])?;

    let bpm = 120;
    let sample_rate = 44100;

    let mut samples = vec![0.0; sample_rate];

    for (index, sample) in kick.into_iter().enumerate() {
        let Some(dest) = samples.get_mut(index) else {
            break;
        };

        *dest += sample;
    }

    for (cymbal_index, cymbals_samples) in cymbals.into_iter().enumerate() {
        let per_beat = 0.5;
        let offset =
            (cymbal_index as f64 * sample_rate as f64 * per_beat * (60.0 / bpm as f64)) as usize;

        for (index, sample) in cymbals_samples.into_iter().enumerate() {
            let Some(dest) = samples.get_mut(offset + index) else {
                break;
            };

            *dest += sample;
        }
    }

    {
        let beat_at = 1.0;
        let offset = (beat_at * sample_rate as f64 * (60.0 / bpm as f64)) as usize;
        for (index, sample) in snare.into_iter().enumerate() {
            let Some(dest) = samples.get_mut(offset + index) else {
                break;
            };

            *dest += sample;
        }
    }

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create("output.wav", spec)?;
    for sample in samples {
        writer.write_sample(sample.clamp(-1.0, 1.0))?;
    }

    Ok(())
}

fn open_wavs(dir: &str) -> Result<Vec<hound::WavReader<std::io::BufReader<std::fs::File>>>> {
    walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().unwrap_or_default() == "wav")
        .map(|e| -> Result<_> {
            Ok(hound::WavReader::open(e.path())
                .map_err(|e| anyhow!("Error reading file: {}", e))?)
        })
        .collect::<Result<Vec<_>>>()
}

fn get_samples(wav: &mut hound::WavReader<std::io::BufReader<std::fs::File>>) -> Result<Vec<f32>> {
    let spec = wav.spec();
    match spec.sample_format {
        hound::SampleFormat::Float => wav
            .samples::<f32>()
            .enumerate()
            .filter(|(i, _)| spec.channels == 1 || i % spec.channels as usize == 0)
            .map(|(_, s)| s.map_err(|e| anyhow!(e)))
            .collect::<Result<Vec<_>>>(),
        hound::SampleFormat::Int => {
            let max = 2_f32.powi(spec.bits_per_sample as i32 - 1);

            wav.samples::<i32>()
                .enumerate()
                .filter(|(i, _)| spec.channels == 1 || i % spec.channels as usize == 0)
                .map(|(_, s)| s.map(|s| s as f32 / max).map_err(|e| anyhow!(e)))
                .collect::<Result<Vec<_>>>()
        }
    }
}

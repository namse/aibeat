use anyhow::*;

fn main() -> Result<()> {
    // 1분짜리 (44100 * 60) 샘플을 만들어보자.
    // 킥만 들어있고, 랜덤한 위치에 15개 들어있다.

    let mut kicks = open_wavs("kicks")?;

    let output_sample_counts = 44100 * 10;
    let mut samples = vec![0.0; output_sample_counts];

    let kick_count = 5;

    let selected_kicks = {
        let mut selected_kicks = Vec::new();
        for _ in 0..kick_count {
            let index: usize = rand::random::<usize>() % kicks.len();
            selected_kicks.push(kicks.remove(index));
        }
        selected_kicks
    };

    let kick_on_sample_timings = (0..kick_count)
        .map(|_| rand::random::<usize>() % output_sample_counts)
        .collect::<Vec<_>>();

    for (mut kick, kick_on_sample_timing) in selected_kicks
        .into_iter()
        .zip(kick_on_sample_timings.iter())
    {
        let kick_samples = trim_sample(&get_samples(&mut kick)?);
        for (index, sample) in kick_samples.into_iter().enumerate() {
            if let Some(dest) = samples.get_mut(kick_on_sample_timing + index) {
                *dest += sample;
            }
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

    std::fs::write(
        "output.txt",
        kick_on_sample_timings
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join("\n"),
    )?;

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

fn trim_sample(samples: &[f32]) -> Vec<f32> {
    let mut start = 0;
    let mut end = samples.len();

    for (index, sample) in samples.iter().enumerate() {
        if *sample != 0.0 {
            start = index;
            break;
        }
    }

    for (index, sample) in samples.iter().enumerate().rev() {
        if *sample != 0.0 {
            end = index;
            break;
        }
    }

    samples[start..end].to_vec()
}

use rand::seq::SliceRandom;
use rayon::prelude::*;
use tch::{
    nn::{Module, OptimizerConfig},
    *,
};

fn main() -> anyhow::Result<()> {
    let input_wav = hound::WavReader::open("../rand-wave/output.wav")
        .unwrap()
        .into_samples()
        .map(|s| s.unwrap())
        .collect::<Vec<f32>>();
    let input_timings = std::fs::read_to_string("../rand-wave/output.txt")?
        .split('\n')
        .map(|s| s.parse::<usize>().unwrap())
        .collect::<Vec<usize>>();

    let epochs = 5;
    let learning_rate = 0.01;

    // let device = Device::cuda_if_available();
    // assert!(device.is_cuda());
    let device = Device::Cpu;

    let var_store = nn::VarStore::new(device);

    let model = MyModelV1::new(&var_store.root(), 441);

    let mut optimizer = nn::Adam::default()
        .build(&var_store, learning_rate)
        .unwrap();

    let data_set = DataSet {
        wav: input_wav,
        timings: input_timings,
    };

    for epoch in 0..epochs {
        println!("epoch: {}", epoch);
        train(&model, data_set.clone(), &mut optimizer, device)?;
        test(&model, data_set.clone(), device);
        var_store.save(format!("var_store-{epoch}"))?;
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct DataSet {
    wav: Vec<f32>,
    timings: Vec<usize>,
}

fn train(
    model: &MyModelV1,
    data_set: DataSet,
    optimizer: &mut nn::Optimizer,
    device: Device,
) -> anyhow::Result<()> {
    let random_start_indexes = {
        let mut indexes = (0..data_set.wav.len() - model.window_size).collect::<Vec<usize>>();
        indexes.shuffle(&mut rand::thread_rng());
        indexes
    };
    let batch_len = random_start_indexes.len();
    let window_size = model.window_size;
    let mut now = std::time::Instant::now();

    let (tx, rx) = std::sync::mpsc::sync_channel(50000);

    std::thread::spawn(move || {
        let wav = data_set.wav;
        let timings = (0..wav.len())
            .map(|sample_index| {
                if data_set.timings.contains(&sample_index) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect::<Vec<f64>>();

        random_start_indexes
            // .chunks(std::thread::available_parallelism().unwrap().get())
            // .collect::<Vec<_>>()
            .into_par_iter()
            // .for_each(|sample_indexes| {
            //     for sample_index in sample_indexes {
            .for_each(|sample_index| {
                let start = sample_index;
                let end = sample_index + window_size;

                let timings = Tensor::from_slice(&timings[start..end]).to_device(device);
                let input = Tensor::from_slice(&wav[start..end]).to_device(device);

                tx.send((input, timings)).unwrap();
                // }
            });
    });

    let mut batch = 0;
    loop {
        batch += 1;
        let (input, timings) = rx.recv()?;
        let output = model.forward(&input);
        let loss = output.cross_entropy_loss(&timings, None::<Tensor>, Reduction::Mean, -100, 0.0);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if batch % 100 == 0 {
            loss.print();
            println!("[{batch}/{batch_len}] - elapsed: {:?}", now.elapsed());
            now = std::time::Instant::now();
        }
    }
}

fn test(model: &MyModelV1, data_set: DataSet, device: Device) {
    // test set: includes timing x 5, not timing x 5

    let in_timings = {
        let mut timings = data_set.timings.clone();
        timings.shuffle(&mut rand::thread_rng());
        timings.truncate(5);
        timings
    };

    let out_timings = {
        let mut timings = vec![];
        for _ in 0..5 {
            loop {
                let timing = rand::random::<usize>() % (data_set.wav.len() - model.window_size);

                if (timing..timing + model.window_size).all(|i| !data_set.timings.contains(&i)) {
                    timings.push(timing);
                    break;
                }
            }
        }
        timings
    };

    let mut test_loss = 0.0;
    let mut correct = 0;

    for timing in [in_timings, out_timings].concat() {
        let wav = data_set.wav[timing..timing + model.window_size].to_vec();
        let timings = (0..wav.len())
            .map(|sample_index| {
                if data_set.timings.contains(&sample_index) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect::<Vec<f64>>();

        let timings = Tensor::from_slice(&timings).to_device(device);
        let input = Tensor::from_slice(&wav).to_device(device);

        let output = model.forward(&input);
        let loss = output.cross_entropy_loss(&timings, None::<Tensor>, Reduction::Mean, -100, 0.0);
        test_loss += loss.double_value(&[]);
        correct += if output.argmax(Some(0), false) == timings {
            1
        } else {
            0
        };
    }

    println!(
        "Test Error: \n Accuracy: {:>0.1}%, Avg loss: {:>8} \n",
        100.0 * correct as f64 / 10.0,
        test_loss / 10.0
    )
}

struct MyModelV1 {
    stack: nn::Sequential,
    window_size: usize,
}

impl MyModelV1 {
    fn new(vs: &nn::Path, window_size: usize) -> MyModelV1 {
        let stack = nn::seq()
            .add(nn::linear(
                vs,
                window_size as i64,
                window_size as i64,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(
                vs,
                window_size as i64,
                window_size as i64,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(
                vs,
                window_size as i64,
                window_size as i64,
                Default::default(),
            ));
        MyModelV1 { stack, window_size }
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        self.stack.forward(input)
    }
}

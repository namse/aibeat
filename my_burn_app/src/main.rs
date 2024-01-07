mod model;

use burn::{
    backend::{wgpu::AutoGraphicsApi, Wgpu},
    config::Config,
    data::{
        dataloader::{batcher::Batcher, DataLoaderBuilder},
        dataset::Dataset,
    },
    module::Module,
    optim::AdamConfig,
    record::{CompactRecorder, Recorder},
    tensor::{
        backend::{AutodiffBackend, Backend},
        Bool, Data, ElementConversion, Int, Tensor,
    },
    train::{
        metric::{AccuracyMetric, LossMetric},
        LearnerBuilder, TrainOutput, TrainStep, ValidStep,
    },
};
use model::{Model, ModelConfig, Output};

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 1000)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 1024)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 0.01)]
    pub learning_rate: f64,
}

pub struct AudioHitBatcher<B: Backend> {
    device: B::Device,
}
impl<B: Backend> AudioHitBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct AudioHitBatch<B: Backend> {
    pub audio_samples: Tensor<B, 1>,
    pub hits: Tensor<B, 1, Bool>,
    pub targets: Tensor<B, 1, Int>,
}

const WINDOW_SIZE: usize = 4410;
const HIT_SIZE: usize = WINDOW_SIZE / 2 - 1;

#[derive(Clone, Copy, Debug)]
struct AudioHitItem {
    audio: [f32; WINDOW_SIZE],
    hit: [bool; WINDOW_SIZE / 2 - 1],
    hit_on_middle: bool,
}

struct AudioHitDataSet {
    items: Vec<AudioHitItem>,
}

impl AudioHitDataSet {
    fn train() -> Self {
        Self {
            items: vec![AudioHitItem {
                audio: [0.0; WINDOW_SIZE],
                hit: [false; HIT_SIZE],
                hit_on_middle: false,
            }],
        }
    }
    fn test() -> Self {
        Self {
            items: vec![AudioHitItem {
                audio: [0.0; WINDOW_SIZE],
                hit: [false; HIT_SIZE],
                hit_on_middle: false,
            }],
        }
    }
}

impl Dataset<AudioHitItem> for AudioHitDataSet {
    fn get(&self, index: usize) -> Option<AudioHitItem> {
        self.items.get(index).copied()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

impl<B: Backend> Batcher<AudioHitItem, AudioHitBatch<B>> for AudioHitBatcher<B> {
    fn batch(&self, items: Vec<AudioHitItem>) -> AudioHitBatch<B> {
        let audio_samples = items
            .iter()
            .map(|item| Data::from(item.audio))
            .map(|data| Tensor::<B, 1>::from_data(data.convert()))
            .collect();

        let hits = items
            .iter()
            .map(|item| Data::from(item.hit))
            .map(|data| Tensor::<B, 1, Bool>::from_data(data))
            .collect();

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data(Data::from([{
                    if item.hit_on_middle { 1 } else { 0 }.elem()
                }]))
            })
            .collect();

        let audio_samples = Tensor::cat(audio_samples, 0).to_device(&self.device);
        let hits = Tensor::cat(hits, 0).to_device(&self.device);
        let targets = Tensor::cat(targets, 0).to_device(&self.device);

        AudioHitBatch {
            audio_samples,
            hits,
            targets,
        }
    }
}
impl<B: AutodiffBackend> TrainStep<AudioHitBatch<B>, Output<B>> for Model<B> {
    fn step(&self, batch: AudioHitBatch<B>) -> TrainOutput<Output<B>> {
        let item = self.forward_output(batch.audio_samples, batch.hits, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<AudioHitBatch<B>, Output<B>> for Model<B> {
    fn step(&self, batch: AudioHitBatch<B>) -> Output<B> {
        self.forward_output(batch.audio_samples, batch.hits, batch.targets)
    }
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    std::fs::create_dir_all(artifact_dir).ok();
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let batcher_train = AudioHitBatcher::<B>::new(device.clone());
    let batcher_valid = AudioHitBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(AudioHitDataSet::train());

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(AudioHitDataSet::test());

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device])
        .num_epochs(config.num_epochs)
        .build(
            config.model.init::<B>(),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}

fn main() {
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;

    let device = burn::backend::wgpu::WgpuDevice::default();

    type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;
    train::<MyAutodiffBackend>(
        "artifact",
        TrainingConfig::new(ModelConfig::new(WINDOW_SIZE, 16), AdamConfig::new()),
        device,
    );

    let device = burn::backend::wgpu::WgpuDevice::default();

    let config =
        TrainingConfig::load("artifact/config.json").expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load("artifact/model".into())
        .expect("Trained model should exist");
    let model = config
        .model
        .init_with::<MyBackend>(record)
        .to_device(&device);

    let audio_samples = Tensor::<MyBackend, 1>::from_data([0.0; WINDOW_SIZE]);
    let hits = Tensor::<MyBackend, 1, Bool>::from_data([false; HIT_SIZE]);

    let output = model.forward(audio_samples, hits);
    println!("output: {:?}", output.into_scalar());
}

use burn::{
    config::Config,
    module::Module,
    nn::{loss::CrossEntropyLoss, Dropout, DropoutConfig, Linear, LinearConfig},
    tensor::{backend::Backend, Bool, Int, Tensor},
    train::metric::{AccuracyInput, Adaptor, LossInput},
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    dropout: Dropout,
    audio_linear1: Linear<B>,
    hit_linear1: Linear<B>,
    merge_linear1: Linear<B>,
    merge_linear2: Linear<B>,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    window_size: usize,
    hidden_size: usize,
    #[config(default = "0.5")]
    dropout: f64,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self) -> Model<B> {
        Model {
            audio_linear1: LinearConfig::new(self.window_size, self.hidden_size).init(),
            hit_linear1: LinearConfig::new(self.window_size / 2 - 1, self.hidden_size).init(),
            merge_linear1: LinearConfig::new(self.hidden_size * 2, self.hidden_size).init(),
            merge_linear2: LinearConfig::new(self.hidden_size, 1).init(),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
    /// Returns the initialized model using the recorded weights.
    pub fn init_with<B: Backend>(&self, record: ModelRecord<B>) -> Model<B> {
        Model {
            audio_linear1: LinearConfig::new(self.window_size, self.hidden_size)
                .init_with(record.audio_linear1),
            hit_linear1: LinearConfig::new(self.window_size / 2 - 1, self.hidden_size)
                .init_with(record.hit_linear1),
            merge_linear1: LinearConfig::new(self.hidden_size * 2, self.hidden_size)
                .init_with(record.merge_linear1),
            merge_linear2: LinearConfig::new(self.hidden_size, 1).init_with(record.merge_linear2),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> Model<B> {
    /// # Shapes
    ///   - Audio Samples <bool> [window_size]
    ///   - hits <bool> [window_size]
    ///   - Output [hit_or_not]
    pub fn forward(&self, audio_samples: Tensor<B, 1>, hits: Tensor<B, 1, Bool>) -> Tensor<B, 2> {
        let audio = {
            let x = self.audio_linear1.forward(audio_samples.reshape([1, -1]));

            self.dropout.forward(x)
        };

        let hit = {
            let x = self.hit_linear1.forward(hits.float().reshape([1, -1]));

            self.dropout.forward(x)
        };

        let merged = Tensor::cat([audio, hit].to_vec(), 1);

        let x = self.merge_linear1.forward(merged);
        let x = self.dropout.forward(x);

        self.merge_linear2.forward(x)
    }
    pub fn forward_output(
        &self,
        audio_samples: Tensor<B, 1>,
        hits: Tensor<B, 1, Bool>,
        targets: Tensor<B, 1, Int>,
    ) -> Output<B> {
        let output = self.forward(audio_samples, hits);
        let loss = CrossEntropyLoss::default().forward(output.clone(), targets.clone());

        Output {
            loss,
            output,
            targets,
        }
    }
}

pub struct Output<B: Backend> {
    /// The loss.
    pub loss: Tensor<B, 1>,

    /// The output.
    pub output: Tensor<B, 2>,

    /// The targets.
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Adaptor<AccuracyInput<B>> for Output<B> {
    fn adapt(&self) -> AccuracyInput<B> {
        AccuracyInput::new(self.output.clone(), self.targets.clone())
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for Output<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

use burn::{
    module::Param,
    prelude::*,
    tensor::activation::{sigmoid, tanh},
};
use serde::{Deserialize, Serialize};

use crate::functions::init_weights::{
    calculate_decay_speed, get_learning_rate_lora_bias, get_value_lora_bias, orthogonal_init,
    zeros_init,
};

#[derive(Config, Debug)]
pub struct LoRAConfig {
    num_cells: usize,
    embedded_dim: usize,
    hidden_dim: usize,
    head_size: usize,

    has_bias: bool,
    activation_fn: ActivationFn,
}

impl LoRAConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LoRA<B> {
        LoRA {
            w_a: Param::from_tensor(Tensor::empty([self.embedded_dim, self.hidden_dim], device)),
            w_b: Param::from_tensor(Tensor::empty([self.hidden_dim, self.embedded_dim], device)),
            bias: if self.has_bias {
                Some(Param::from_tensor(Tensor::empty(
                    [1, 1, self.embedded_dim],
                    device,
                )))
            } else {
                None
            },
            activation_fn: self.activation_fn.clone(),

            num_cells: self.num_cells,
            embedded_dim: self.embedded_dim,
            head_size: self.head_size,
        }
    }
}

#[derive(Module, Debug)]

pub struct LoRA<B: Backend> {
    pub w_a: Param<Tensor<B, 2>>,
    pub w_b: Param<Tensor<B, 2>>,
    pub bias: Option<Param<Tensor<B, 3>>>,
    #[module(skip)]
    activation_fn: ActivationFn,

    num_cells: usize,
    embedded_dim: usize,
    head_size: usize,
}

impl<B: Backend> LoRA<B> {
    pub fn init_weight(&mut self, cell_id: usize, lora_type: LoRAType, device: &B::Device) {
        zeros_init(&mut self.w_a);

        orthogonal_init(&mut self.w_b, Some(0.1));

        if let Some(ref mut bias) = self.bias {
            match lora_type {
                LoRAType::WeightDecay => {
                    *bias = Param::from_tensor(
                        calculate_decay_speed(
                            self.num_cells,
                            self.embedded_dim,
                            self.head_size,
                            cell_id,
                            device,
                        )
                        .reshape([1, 1, self.embedded_dim]),
                    );
                }
                LoRAType::LearningRate => {
                    *bias = Param::from_tensor(
                        get_learning_rate_lora_bias(self.embedded_dim, self.head_size, device)
                            .reshape([1, 1, self.embedded_dim]),
                    );
                }
                LoRAType::ValueResidual => {
                    *bias = Param::from_tensor(
                        get_value_lora_bias(self.embedded_dim, device).reshape([
                            1,
                            1,
                            self.embedded_dim,
                        ]),
                    );
                }
            }
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Group view operations before compute block for better fusion
        let w_a = self.w_a.val().unsqueeze_dim(0);
        let w_b = self.w_b.val().unsqueeze_dim(0);

        let x = x.matmul(w_a);

        let x = match self.activation_fn {
            ActivationFn::Sigmoid => sigmoid(x),
            ActivationFn::Tanh => tanh(x),
            ActivationFn::NoOP => x,
        };

        let x = x.matmul(w_b);

        match &self.bias {
            Some(bias) => bias.val() + x,
            None => x,
        }
    }

    pub fn forward_2d(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = x.matmul(self.w_a.val());

        let x = match self.activation_fn {
            ActivationFn::Sigmoid => sigmoid(x),
            ActivationFn::Tanh => tanh(x),
            ActivationFn::NoOP => x,
        };

        let x = x.matmul(self.w_b.val());

        match &self.bias {
            Some(bias) => {
                let bias: Tensor<B, 1> = bias.val().squeeze_dim::<2>(0).squeeze_dim::<1>(0);
                bias.unsqueeze_dim(0) + x
            }
            None => x,
        }
    }
}

#[derive(Module, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]

pub enum ActivationFn {
    Sigmoid,
    Tanh,
    NoOP,
}

#[derive(Clone, Debug, Eq, PartialEq)]

pub enum LoRAType {
    WeightDecay,
    LearningRate,
    ValueResidual,
}

pub struct LoRARanks {
    pub min_d_model: usize,
    pub weight_decay_lora: usize,
    pub learning_rate_lora: usize,
    pub value_residual_lora: usize,
    pub output_gate_lora: usize,
}

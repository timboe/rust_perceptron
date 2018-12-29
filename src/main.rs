extern crate mnist;
extern crate rand;
//#[macro_use]
extern crate rulinalg;

use mnist::{Mnist, MnistBuilder};
use rulinalg::matrix::{BaseMatrix, Matrix};
use rulinalg::vector::Vector;
use rand::prelude::*;
use utils::math::sigmoid;
// use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};


struct Network {
  layers: Vec<Layer>,
  connections: Vec<Connections>,
  input_size: usize
}

struct Response {
  layers: Vec<Layer>,
  input_size: usize
}

struct Layer {
  vertexes: Vector<f32>
}

struct Connections {
  edges: Matrix<f32>
}

impl Response {
  pub fn new(network: &Network) -> Response {
    let mut layers: Vec<Layer> = Vec::new();
    let input_size = network.input_size;
    for l in network.layers() { // Copy vertexes structure, don't care for edges
      layers.push( Layer::new( l.size() ) );
    }
    Response {
      layers,
      input_size
    }
  }

  // pub fn yield_layer(&mut self, index : usize) -> Vector<f32> {
  //   self.layers.remove(index).vertexes
  // }

  // pub fn set_layer(&mut self, index : usize, v : Vector<f32>) {
  //   self.layers[index].vertexes = v;
  // }


  pub fn get_layer(&self, index : usize) -> &Vector<f32> {
    & (self.layers[index].vertexes)
  }

  pub fn get_layer_mut(&mut self, index : usize) -> &mut Vector<f32> {
    &mut (self.layers[index].vertexes)
  }

  pub fn get_output_layer(&self) -> &Vector<f32> {
    & (self.layers[ self.layers.len() - 1 ].vertexes)
  }

  pub fn set_input(&mut self, entry: usize, input: &Vec<u8>) {
    let mut index: usize = 0;
    let vertexes = self.layers[0].as_mut();
    for i in (entry*self.input_size)..(entry+1)*self.input_size {
      let value: f32 = (*input.get(i).unwrap()) as f32 / u8::max_value() as f32;
      vertexes[index] = value;
      index += 1;
    }
  }
}

impl Network {
  pub fn new() -> Network {
    Network{
      layers: Vec::new(),
      connections: Vec::new(),
      input_size: 0
    }
  }

  pub fn add_layer(&mut self, layer_size: usize) {
    self.layers.push( Layer::new(layer_size) );
    let length = self.layers.len();
    if length == 1 {
      self.input_size = layer_size;
      println!("Input layer: size {}", layer_size);
      return
    }
    // Make connections between prev layer and this one
    let previous_layer_size = self.layers.get(length - 2).unwrap().size();
    self.connections.push( Connections::new(layer_size, previous_layer_size) );
    println!("Inner layer: size {}, connections: {}x{}={}", layer_size, layer_size, previous_layer_size, previous_layer_size * layer_size);
  }

  pub fn get_layer(&self, index : usize) -> &Vector<f32> {
    & (self.layers[index].vertexes)
  }

  pub fn get_connections(&self, index : usize) -> &Matrix<f32> {
    & (self.connections[index].edges)
  }

  pub fn layers(&self) -> &Vec<Layer> {
    &(self.layers)
  }

  pub fn randomize(&mut self, rnd: &mut ThreadRng) {
    for l in self.layers.as_mut_slice() {
      for cell in l.as_mut().mut_data() {
        *cell = rnd.gen::<f32>();
      }
    }
    for l in self.connections.as_mut_slice() {
      for cell in l.as_mut().mut_data() {
        *cell = rnd.gen::<f32>();
      }
    }
  }

  pub fn apply(&self, response: &mut Response, answer: &Vector<f32>) -> f32 {
    // walk through the network
    for (index, vertex) in self.layers.iter().enumerate() {
      if index == 0 { continue; } // This is our set of inputs

      // let new_matrix = self.get_connections(index-1) * response.get_layer(index-1);
      (*response.get_layer_mut(index)) = self.get_connections(index-1) * response.get_layer(index-1);

      println!("connections matrix size {}x{} \n{}", self.get_connections(index-1).rows(), self.get_connections(index-1).cols(), self.get_connections(index-1));
      println!("layers vector size {} \n{}", response.get_layer(index-1).size(), response.get_layer(index-1));
      println!("result vector (before bias) size {} \n{}", response.get_layer(index).size(), response.get_layer(index));

      (*response.get_layer_mut(index)) -= self.get_layer(index); // subtract biases

      println!("result vector (after bias) size {} \n{}", response.get_layer(index).size(), response.get_layer(index));

      for element in response.get_layer_mut(index).iter_mut() { *element = sigmoid(*element); }

      println!("result vector (after sigmoid) size {} \n{}", response.get_layer(index).size(), response.get_layer(index));

    }

    // Compute distane from answer
    let mut cost: f32 = 0.;
    let mut iter = response.get_output_layer().iter().zip( answer.iter() );

    println!("correct answer is \n{}", answer);
    for (resp, ans) in iter {
      cost += f32::powf(*resp - *ans, 2.0);
    }
    println!("cost is {}", cost);
    cost
  }
}

impl Layer {
  pub fn new(layer_size: usize) -> Layer {
    Layer {
      vertexes: Vector::new( vec![0f32; layer_size] )
    }
  }
  pub fn size(&self) -> usize {
    self.vertexes.size()
  }
  pub fn as_ref(&self) -> &Vector<f32> {
    &(self.vertexes)
  }
  pub fn as_mut(&mut self) -> &mut Vector<f32> {
    &mut(self.vertexes)
  }
}

impl Connections {
  pub fn new(previous_layer_size: usize, current_size: usize) -> Connections {
    let cells = previous_layer_size * current_size;
    Connections {
      edges: Matrix::new( previous_layer_size, current_size, vec![0f32; cells] )
    }
  }
  pub fn as_ref(&self) -> &Matrix<f32> {
    &(self.edges)
  }
  pub fn as_mut(&mut self) -> &mut Matrix<f32> {
    &mut(self.edges)
  }
}

fn main() {
    let trn_size: usize = 50000;
    let rows: usize = 28;
    let cols = rows;

    // Deconstruct the returned Mnist struct.
    let Mnist { trn_img, trn_lbl, .. } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(trn_size as u32)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let mut answers = vec![ Vector::new( vec![0f32; 10] ) ; 10];
    for (index, element) in answers.iter_mut().enumerate() {
      // We set the index'th elemnt to 1
      element[index] = 1.;
      println!("The answer vector for {} is \n{}", index, element);
    }

    // Convert the flattened training images vector to a matrix.
    let trn_img_matrix = Matrix::new(trn_size * rows, cols, trn_img.clone());    

    let mut network = Network::new();
    network.add_layer(rows * cols);
    network.add_layer(16);
    network.add_layer(16);
    network.add_layer(10);

    let mut rnd_gen = thread_rng();
    network.randomize(&mut rnd_gen);

    let mut response = Response::new(&network);

    for i in 0..1 {
        // Get the label of the digit.
        let label: usize = trn_lbl[i] as usize;
        if i < 5 {
         
          println!("The digit is a {}.", label);

          let row_indexes = ( rows*i .. rows*(i+1) ).collect::<Vec<_>>(); 

          // Get the image of the first digit.
          let image = trn_img_matrix.select_rows(&row_indexes);
          println!("The image looks like... \n{}", image);
        }

        response.set_input(i, &trn_img);

        let cost = network.apply(&mut response, answers.get(label).unwrap());

        // Convert the training images to f32 values scaled between 0 and 1.
        // let trn_img: Matrix<f32> = trn_img.try_into().unwrap() / 255.0;

        // Get the image of the first digit and round the values to the nearest tenth.
        // let first_image = trn_img.select_rows(&row_indexes)
            // .apply(&|p| (p * 10.0).round() / 10.0);
        // println!("The image looks like... \n{}", first_image);
    }


}
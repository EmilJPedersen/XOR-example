var tf = require('@tensorflow/tfjs');

async function go() 
{
  // Creating a model to predict the output
  const model = tf.sequential();
  model.add(tf.layers.dense({units: 8, activation: 'tanh', inputShape: [2]}));
  model.add(tf.layers.dense({units: 1}));

  model.compile({loss: 'meanSquaredError', optimizer: 'rmsprop'});

  // Creating dataset
  const training_data = tf.tensor2d([[0,0],[0,1],[1,0],[1,1]]);
  const target_data = tf.tensor2d([[0],[1],[1],[0]]);

  // Train the model
  for (let i = 1; i < 100 ; ++i) 
  {
    var h = await model.fit(training_data, target_data, {epochs: 30});
      console.log("Loss after Epoch " + i + " : " + h.history.loss[0]);
  }

  // Test the model and display output 
  model.predict(training_data).print();
  }
go();


// Chord Bot

import * as tf from '@tensorflow/tfjs'
import * as aFancyBrain from '@tensorflow-models/speech-commands'

import * as UI from 'UI'

const INPUT_SHAPE = [3, 232, 1]
const robot = {
  brain: null,
  knowledge: [],
  ears: aFancyBrain.create('BROWSER_FFT')
}

robot.wakeUp = async function() {
  
  // Create a sequential model so we can stack layers
  robot.brain = tf.sequential()

  // Convolutional neural network layer 
  // spliting each input into n neurons
  robot.brain.add(tf.layers.depthwiseConv2d({ ... }))

  // "Downsampling" layer
  robot.brain.add(tf.layers.maxPooling2d({ ... }))

  // Flatten so it fits the output layer 
  robot.brain.add(tf.layers.flatten())

  // Add another some dense CNN layers (all neurons are connected)
  robot.brain.add(tf.layers.dense({ ... }))

  // Create the model
  robot.brain.compile({ ... })

  UI.update()
},

robot.learn = function (type) {
  robot.ears.listen(async ({ spectrogram }) => {
    const { frameSize, data } = spectrogram
    
    // Normalize audio sample
    const audioWaves = normalizeAudio(data, frameSize)

    robot.knowledge.push({ type, audioWaves })

  })
},

robot.think = async function() {
  const ys = tf.oneHot(robot.knowledge.map(({ type }) => type), 3)
  const flattened = flattenAudioSamples(robot.knowledge)
  const xs = tf.tensor(flattened, [robot.knowledge.length, ...INPUT_SHAPE])

  await robot.brain.fit(xs, ys, { ... })

  tf.dispose([xs, ys]) 

},

robot.listen = function() {

  robot.ears.listen(async ({ spectrogram }) => {
    const { frameSize, data } = spectrogram

    // Normalize audio sample
    const audioWaves = normalizeAudio(data, frameSize)
    
    // Convert audioWaves into a tensor
    const whatRobotHeard = tf.tensor(audioWaves, [1, ...INPUT_SHAPE])
    const whatRobotPredicts = robot.brain.predict(whatRobotHeard)

    await UI.update(whatRobotPredicts)

    tf.dispose([audioWaves, whatRobotHeard, whatRobotPredicts])

  })

},




// Chord Bot

import * as tf from '@tensorflow/tfjs'
import * as speechCommands from '@tensorflow-models/speech-commands'

import { 
  byId, 
  onClick, 
  toggleAllButtons, 
  setStatus, 
  flattenAudioSamples, 
  normalizeAudio 
}  from '../utils'

const NUM_FRAMES = 3
const INPUT_SHAPE = [NUM_FRAMES, 232, 1]
const earOptions = {
 overlapFactor: 0.999,
 includeSpectrogram: true,
 invokeCallbackOnNoiseAndUnknown: true
}

const robot = {
  brain: null,
  knowledge: [],
  ears: speechCommands.create('BROWSER_FFT'),

  wakeUp: async function() {
    await robot.ears.ensureModelLoaded()

    robot.brain = tf.sequential()
    
    robot.brain.add(tf.layers.depthwiseConv2d({ 
     depthMultiplier: 8,
     kernelSize: [NUM_FRAMES, 4],
     activation: 'relu',
     inputShape: INPUT_SHAPE }))


    robot.brain.add(tf.layers.maxPooling2d({ 
      poolSize: [1, 2], 
      strides: [2, 2]  }))


    robot.brain.add(tf.layers.flatten())


    robot.brain.add(tf.layers.dense({ units: 4, activation: 'softmax' }))
    

    robot.brain.compile({
     optimizer: tf.train.adam(0.01),
     loss: 'categoricalCrossentropy',
     metrics: ['accuracy'] 
    })

    robot.initiateBody()
  },



  learn: function (type) {
    if (Number.isInteger(type)) {
      robot.ears.listen(async ({ spectrogram }) => {
        const { frameSize, data } = spectrogram

        const audioWaves = normalizeAudio(data.subarray(-frameSize * NUM_FRAMES))

        robot.knowledge.push({ type, audioWaves })
       
        byId('status').textContent = `${robot.knowledge.length} examples collected`;

      }, earOptions)
      setStatus('learning')
    }
    else {
      robot.ears.stopListening()
      setStatus('waiting') 
    }
  },



  think: async function() {
    const ys = tf.oneHot(robot.knowledge.map(({ type }) => type), 4)
    const flattened = flattenAudioSamples(robot.knowledge.map(({ audioWaves }) => audioWaves))
    const xs = tf.tensor(flattened, [robot.knowledge.length, ...INPUT_SHAPE])

    toggleAllButtons(false)
    setStatus('thinking')
    
    await robot.brain.fit(xs, ys, {
      batchSize: 16,
      epochs: 20,
      callbacks: {
       onEpochEnd: robot.updateTrainingStats
      }
    })

    tf.dispose([xs, ys]) 

    setStatus('waiting')
    toggleAllButtons(true)
  },



  listen: function() {
    if (robot.ears.isListening() === false) {
      setStatus('listening')

      robot.ears.listen(async ({ spectrogram }) => {
        const { frameSize, data } = spectrogram

        const audioWaves = normalizeAudio(data.subarray(-frameSize * NUM_FRAMES))
        
        const whatRobotHeard = tf.tensor(audioWaves, [1, ...INPUT_SHAPE])
        const whatRobotPredicts = robot.brain.predict(whatRobotHeard)

        await robot.updateBody(whatRobotPredicts)

        tf.dispose([audioWaves, whatRobotHeard, whatRobotPredicts])

      }, earOptions)
    }
    else {
      robot.ears.stopListening()
      setStatus('waiting')
    }
  },


  toggleLearning: function(label) {
    const toggle = !robot.buttons[label].toggle

    toggleAllButtons(!toggle)

    robot.buttons[label].toggle = toggle
    robot.buttons[label].setAttribute('disabled', false)

    robot.learn(toggle ? label : false)
  },
  updateBody: async function (labelTensors) {
    const labels = await labelTensors.data()

    labels.map((confidence, label) => {
      robot.buttons[label].style.opacity = Math.min(confidence + 0.1, 1);
      if (confidence > 0.8) status = document.body.setAttribute('chord', label)
    })
  },
  updateTrainingStats: function (epoch, logs) {
    byId('status').textContent =
      `Epoch: ${epoch + 1}\n\nAccuracy: ${(logs.acc * 100).toFixed(2)}%`;
  },
  initiateBody: function() {
    onClick(robot.buttons["0"], () => robot.toggleLearning(0))
    onClick(robot.buttons["1"], () => robot.toggleLearning(1))
    onClick(robot.buttons["2"], () => robot.toggleLearning(2))
    onClick(robot.buttons["3"], () => robot.toggleLearning(3))

    onClick(robot.buttons.think, robot.think)
    onClick(robot.buttons.listen, robot.listen)
    
    setStatus('waiting')
  },
  buttons: {
    
    0: byId('chord-1'),
    1: byId('chord-2'),
    2: byId('chord-3'),
    3: byId('noise'),

    think: byId('think'),
    listen: byId('listen'),
  }
}

onClick(byId('on-button'), robot.wakeUp)


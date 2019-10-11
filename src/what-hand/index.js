
// What Hand Example

import * as tf from '@tensorflow/tfjs'

import { byId, onClick, toggleAllButtons, setStatus }  from '../utils'

const NR_OF_TOUCH_EVENTS = 100
const INPUT_SHAPE = [NR_OF_TOUCH_EVENTS, 1]

const robot = {
  brain: null,
  knowledge: [],


  wakeUp: async function() {
    // Create a sequential model so we can stack layers
    robot.brain = tf.sequential()
     
    // A simple input layer
    robot.brain.add(tf.layers.dense({ inputShape: INPUT_SHAPE, units: 3 }))
    
    // Flattens the input layer to fit the output layer
    robot.brain.add(tf.layers.flatten())

    // Output layer 
    robot.brain.add(tf.layers.dense({ units: 2 }))
    
    // Compile our Model
    robot.brain.compile({
      optimizer: tf.train.adam(), 
      loss: tf.losses.meanSquaredError,
      metrics: ['mse', 'accuracy'],
    })

    robot.initiateBody()
  },


  learn: function (type) {
    if (Number.isInteger(type)) {
      let touchEvents = []
      setStatus('learning')
      
      document.onmousemove = ({ clientX, clientY }) => {
        const normalizedX = (clientX/window.innerWidth) - 0.5

        touchEvents.push([normalizedX])  

        if (touchEvents.length === NR_OF_TOUCH_EVENTS) {
          robot.knowledge.push({type, value: touchEvents })
          touchEvents = []
        }

        byId('status').textContent = 
          `${robot.knowledge.length} examples collected (${touchEvents.length} / ${NR_OF_TOUCH_EVENTS})`
      }
    }
    else {
      setStatus('waiting') 
      document.onmousemove = null
    }
  },



  think: async function() {
    const ys = tf.oneHot(robot.knowledge.map(e => e.type), 2)
    const values = robot.knowledge.map(({ value }) => value)
    
    const xs = tf.tensor(values, [robot.knowledge.length, ...INPUT_SHAPE] )
    
    toggleAllButtons(false)

    setStatus('thinking')

    await robot.brain.fit(xs, ys, {
      batchSize: NR_OF_TOUCH_EVENTS/2,
      epochs: 133,
      callbacks: {
       onEpochEnd: robot.updateTrainingStats
      }
    })

    tf.dispose([xs, ys]) 

    setStatus('waiting')

    toggleAllButtons(true)
  },



  listen: function() {
    let touchEvents = []
    
    document.onmousemove = async function({ clientX, clientY }) {
      // normalize x value between -0.5 - 0.5
      const normalizedX = (clientX/window.innerWidth) - 0.5

      touchEvents.push([normalizedX])

      if (touchEvents.length === NR_OF_TOUCH_EVENTS) {
        const whatRobotThinks = tf.tensor([touchEvents], [1, ...INPUT_SHAPE] )
        const whatRobotPredicts = robot.brain.predict(whatRobotThinks)

        await robot.updateBody(whatRobotPredicts)

        touchEvents = []

        tf.dispose([whatRobotThinks, whatRobotPredicts])
      }
    }

    setStatus('listening')
  },

  
  // UI Stuff Below. We can ignore those. :)
  toggleLearning: function(label) {
    const toggle = !robot.buttons[label].toggle

    toggleAllButtons(!toggle)

    robot.buttons[label].toggle = toggle
    robot.buttons[label].setAttribute('disabled', false)

    robot.learn(toggle ? label : false)
  },
  updateBody: async function (predictions) {
    const labels = await predictions.data()

    labels.map((confidence, label) => {
      robot.buttons[label].style.opacity = Math.min(confidence + 0.1, 1)
      if (confidence > 0.8) document.body.setAttribute('what-hand', label)
    })
  },
  updateTrainingStats: function (epoch, logs) {
    byId('status').textContent =
      `Epoch: ${epoch + 1}\n\nAccuracy: ${(logs.acc * 100).toFixed(2)}%`
  },
  initiateBody: function() {
    onClick(robot.buttons["0"], () => robot.toggleLearning(0))
    onClick(robot.buttons["1"], () => robot.toggleLearning(1))

    onClick(robot.buttons.think, robot.think)
    onClick(robot.buttons.listen, robot.listen)
    
    setStatus('waiting')
  },
  buttons: {
    
    0: byId('chord-1'),
    1: byId('chord-2'),

    think: byId('think'),
    listen: byId('listen'),
  }
}

onClick(byId('on-button'), robot.wakeUp)
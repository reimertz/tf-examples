
// What Hand Bot pseudo code

import * as tf from '@tensorflow/tfjs'
import * as UI from 'UI'

const NR_OF_TOUCH_EVENTS = 100
const INPUT_SHAPE = [NR_OF_TOUCH_EVENTS, 1]

const robot = {
  brain: null,
  knowledge: []
}

robot.wakeUp = function() {

	robot.brain = tf.sequential()

	// Input layer
	robot.brain.add(
    robot.brain.add(tf.layers.dense({ inputShape: INPUT_SHAPE, units: 3 }))
	)

	robot.brain.add(
		tf.layers.flatten()
	)

	// Output layer
  robot.brain.add(tf.layers.dense({ units: 2 }))

	UI.update()
}

robot.learn = function ( type ) { 
  let touchEvents = []
  
  document.onmousemove = ({ clientX }) => {
  	// normalize touch events between -0.5 - 0.5
    const normalizedX = (clientX/window.innerWidth) - 0.5

    touchEvents.push([normalizedX])  

    if (touchEvents.length === NR_OF_TOUCH_EVENTS) {
      robot.knowledge.push({type, value: touchEvents })
      touchEvents = []
    }

    UI.update()
  }
}

robot.think = async function() {

  const ys = tf.oneHot(robot.knowledge.map(e => e.type), 2)
  const values = robot.knowledge.map(({ value }) => value)  
  const xs = tf.tensor(values, [robot.knowledge.length, ...INPUT_SHAPE])

  await robot.brain.fit(xs, ys, { /* some fancy arguments */ })

  tf.dispose([xs, ys])

  UI.update()
}

robot.listen = function() {
  let touchEvents = []
  
  document.onmousemove = async function({ clientX }) {
    const normalizedX = (clientX/window.innerWidth) - 0.5

    touchEvents.push([normalizedX])

    if (touchEvents.length === NR_OF_TOUCH_EVENTS) {
      const whatRobotThinks = tf.tensor([touchEvents], [1, ...INPUT_SHAPE])
      const whatRobotPredicts = robot.brain.predict(whatRobotThinks)

      await UI.update(whatRobotPredicts)

      touchEvents = []

      tf.dispose([whatRobotThinks, whatRobotPredicts])
    }
  }
}
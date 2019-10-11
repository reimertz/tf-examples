// Support App

import * as tf from '@tensorflow/tfjs'
import * as aFancyBrain from '@tensorflow-models/mobilenet'
import * as knnClassifier from '@tensorflow-models/knn-classifier'

import * as UI from 'UI'

const robot = {
  brain: null,
  knowledge: null,
  eyes: byId('webcam')
}

robot.wakeUp = async function() {
  
  // Load MobileNet
  robot.brain = await aFancyBrain.load()

  // Load KNN classifier
  robot.knowledge = knnClassifier.create()

  robot.work()
}


robot.work = async function() {
  if (robot.knowledge.getNumClasses() > 0) {
    
    const stateOfBrain = robot.brain.infer(robot.eyes, 'conv_preds')
    const robotPredicts = await robot.knowledge.predictClass(stateOfBrain)
    
    UI.update(robotPredicts)

    tf.dispose([stateOfBrain, robotPredicts])
  }

  window.requestAnimationFrame(() => robot.work())
}

robot.explain = function (explaination) {
  // Get "brain state" (intermediate activation)
  const stateOfBrain = robot.brain.infer(robot.eyes, 'conv_preds')

  // Explain to the robot what it sees. 
  robot.knowledge.addExample(stateOfBrain, explaination)
}


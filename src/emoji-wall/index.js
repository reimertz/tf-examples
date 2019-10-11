
import * as tf from '@tensorflow/tfjs'
import * as faceapi from 'face-api.js';

import { byId, initiateWebcam }  from '../utils'
import { getAPI }  from '../firebase'

function expressionsToEmoji(expressions) {
  const [emotion, probability] = Object.entries(expressions)
    .sort(([emotion1, prob1], [emotion2, prob2])  => prob2 - prob1)
    .filter(([emotion, prob]) => prob >= 0.7)[0] || []

  switch(emotion) {
    case 'neutral':
      return '😐'
    case 'happy':
      return '😍'
    case 'sad':
      return '😢'
    case 'angry':
      return '😡'
    case 'fearful':
      return '😨'
    case 'disgusted':
      return '🤮'
    case 'surprised':
      return '😱'
    default:
      return null
  }
}

const robot = {
  api: null, 
  eyes: byId('webcam'),
  mood: '🤖',

  wakeUp: async function() {
    try { 
      await initiateWebcam(robot.eyes) 
    }
    catch (e) {
      alert(`Couldn't access your camera 😢`);
    }
    
    await faceapi.nets.ssdMobilenetv1.load('models/face-api')
    await faceapi.nets.faceExpressionNet.load('models/face-api')

    robot.api = await getAPI()
    robot.work()
  },

  work: async function() {
    try {
      const { expressions } = await faceapi.detectSingleFace(robot.eyes).withFaceExpressions()
      const oldEmoji = robot.mood
      robot.mood = (expressionsToEmoji(expressions) || robot.mood)
       
      if (robot.mood !== oldEmoji) {
        byId('emoji').textContent = robot.mood
        robot.api.setEmoji(robot.mood)
      }

    }
    catch (e) { console.log(e) }

    window.requestAnimationFrame(() => robot.work())
  }
}

robot.wakeUp()


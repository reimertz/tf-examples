
// Support App

import * as tf from '@tensorflow/tfjs'

import * as aFancyBrain from '@tensorflow-models/mobilenet'
import * as knowledgeCreator from '@tensorflow-models/knn-classifier'

import { byId, onClick, initiateWebcam, setStatus }  from '../utils'

const robot = {
  brain: null,
  knowledge: null,
  eyes: byId('webcam'),


  wakeUp: async function() {
    // Load MobileNet
    robot.brain = await aFancyBrain.load()

    // Load KNN classifier
    robot.knowledge = knowledgeCreator.create();

    robot.initiateBody()
    robot.work()
  },


  explain: function (explaination) {
    // Get "brain state" (intermediate activation)
    const stateOfBrain = robot.brain.infer(robot.eyes, 'conv_preds')

    // Explain to the robot what they see. 
    robot.knowledge.addExample(stateOfBrain, explaination)
  },


  work: async function() {
    if (robot.knowledge.getNumClasses() > 0) {

      const stateOfBrain = robot.brain.infer(robot.eyes, 'conv_preds')
      const robotPredicts = await robot.knowledge.predictClass(stateOfBrain)
      
      robot.updateBody(robotPredicts)
    }

    window.requestAnimationFrame(() => robot.work())
  },


  // UI Stuff Below. We can ignore those. :)
  initiateBody: function() { 
    setStatus('working')
    onClick(robot.buttons.charging, () => robot.explain('charging'))
    onClick(robot.buttons.notCharging, () => robot.explain('notCharging'))         
  },
  updateBody: async function (predictions) {
    Object.entries(predictions.confidences).map(([key, confidence]) => {
      robot.buttons[key].style.opacity = confidence + 0.25;
      if (confidence > 0.8) status = document.body.setAttribute('charging', key)
    })
  },
  buttons: {
    charging: byId('charging'),
    notCharging: byId('not-charging')
  },
}





onClick(byId('on-button'), robot.wakeUp)

const button = document.getElementById('button');
const select = document.getElementById('select');


function stopMediaTracks(stream) {
  stream.getTracks().forEach(track => {
    track.stop();
  });
}

function gotDevices(mediaDevices) {
  select.innerHTML = '';
  select.appendChild(document.createElement('option'));
  let count = 1;
  mediaDevices.forEach(mediaDevice => {
    if (mediaDevice.kind === 'videoinput') {
      const option = document.createElement('option');
      option.value = mediaDevice.deviceId;
      const label = mediaDevice.label || `Camera ${count++}`;
      const textNode = document.createTextNode(label);
      option.appendChild(textNode);
      select.appendChild(option);
    }
  });
}
let currentStream;

button.addEventListener('click', event => {
  if (typeof currentStream !== 'undefined') {
    stopMediaTracks(currentStream);
  }
  const videoConstraints = {};
  if (select.value === '') {
    videoConstraints.facingMode = 'environment';
  } else {
    videoConstraints.deviceId = { exact: select.value };
  }
  const constraints = {
    video: videoConstraints,
    audio: false
  };
  navigator.mediaDevices
    .getUserMedia(constraints)
    .then(stream => {
      currentStream = stream;
      robot.eyes.srcObject = stream;
      return navigator.mediaDevices.enumerateDevices();
    })
    .then(gotDevices)
    .catch(error => {
      console.error(error);
    });
});

navigator.mediaDevices.enumerateDevices().then(gotDevices);




import * as tf from '@tensorflow/tfjs'
import * as faceapi from 'face-api.js';

import { byId }  from '../utils'
import { getAPI }  from '../firebase'

async function app() {
  const api = await getAPI()

  api.getEmojiStream(docs => {
    const emojis = []
    
    docs.forEach((doc, index) => {
      emojis.push(doc.data().emoji)
    })

    byId('emojis').innerHTML = emojis.join(' ')
    
  })
}

app()

